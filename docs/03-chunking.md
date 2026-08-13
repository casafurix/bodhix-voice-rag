# 03 — Chunking and indexing

> The brief: *"Chunking strategy should be vast — don't submit a single naive fixed-size chunking
> approach. We want to see real thought put into how the dataset is split, indexed, and
> retrieved."*

This is the most explicitly graded requirement, so it gets the most engineering.

## The stance

Chunking is an **empirical question with a labelled answer available**, not a matter of taste. The
dataset ships `is_selected` relevance labels, so we can measure every strategy and publish a
leaderboard. Our submission does not claim a thoughtful chunking strategy — it demonstrates twelve
of them behind one interface, shows the ablation table, and ships the measured champion.

Two things follow, and both are stated because they cut against the usual RAG reflex:

**1. On this corpus, fixed-size splitting is not merely naive — it is destructive.** MS MARCO
passages are already coherent 50–120 word web snippets. A 512-token window with 64-token overlap
would *merge unrelated passages* and *cut coherent ones mid-argument*. We implement it anyway, as
the control condition, precisely so the ablation table can show it losing.

**2. The interesting axis is not "how do I split long text" but "what is the retrieval unit, and
what context do I return around it."** Those are two different questions, and conflating them is
the most common RAG mistake. Small units retrieve precisely; large units generate well. Several
strategies below exist purely to decouple the two.

---

## Design axes

Every strategy below is a point in this space. Naming the axes makes the twelve strategies look
like a designed sweep rather than a grab bag.

| Axis | Options |
|---|---|
| **Boundary rule** | fixed tokens · sentence · passage · semantic drift · LLM proposition |
| **Unit size** | sub-passage · passage · multi-passage window |
| **Overlap** | none · fixed stride · sentence-neighbour · hierarchical containment |
| **Retrieve vs return** | same unit · small-to-big · window expansion |
| **Embedding context** | chunk-isolated · doc-conditioned (late chunking) · prefixed (contextual) |
| **Vectors per unit** | one · many (doc2query, multi-view) |
| **Language handling** | per-language · script-aware sizing · cross-lingual twins |

---

## The twelve strategies

Each has an ID used in the ablation table, an implementation note, the hypothesis it tests, and
the cost it carries. All implement one interface:

```python
class Chunker(Protocol):
    id: str
    def chunk(self, doc: PassageDoc) -> list[Chunk]: ...
    # Chunk carries text, char_span, parent_id, extra payload, and
    # optionally its own embedding text (may differ from returned text)
```

That last clause is the important one: **the text we embed and the text we return need not be the
same string.** Half the strategies below exploit that.

---

### S1 · `fixed_512_64` — fixed-size control

512 tokens, 64-token stride overlap, hard cut at token boundaries.

- **Hypothesis:** will underperform, because it violates passage boundaries.
- **Purpose:** the control the brief warns against. Required for the ablation to mean anything.
- **Cost:** trivial.

### S2 · `passage_native` — atomic passage

One passage = one chunk. No splitting at all.

- **Hypothesis:** a strong baseline that many fancier strategies will fail to beat, because MS
  MARCO passages are already well-formed retrieval units.
- **Purpose:** the honest baseline. If nothing beats S2 we say so — a submission that reports
  "our sophisticated strategies did not beat the simple one, here is the evidence" is more
  credible than one that quietly ships the complex loser.
- **Cost:** trivial.

### S3 · `sentence_window` — sentence index, window return

Index individual sentences. On retrieval, return the hit sentence plus ±k neighbouring sentences
from the parent passage.

- **Hypothesis:** small units maximise embedding precision (one sentence = one idea, so the
  vector is not diluted), while the returned window preserves the context the answer needs.
- **Note:** sentence segmentation for Indic scripts needs care — the Devanagari danda `।`, and
  Urdu's `۔`, are not handled by default English splitters. We use `indic-nlp-library` /
  `pysbd`-style rules per script rather than pretending `.` is universal. This is the kind of
  detail that separates a real multilingual pipeline from an English one with a language flag.
- **Cost:** 3–6× more vectors than S2. Window assembly happens **at query time** and is counted
  against `t_core`.

### S4 · `semantic_drift` — embedding-breakpoint splitting

Embed consecutive sentences, compute cosine distance between neighbours, split where distance
exceeds the 95th percentile of the document's own distance distribution.

- **Hypothesis:** boundaries placed at genuine topic shifts beat boundaries placed at arbitrary
  token counts.
- **Note:** on short MS MARCO passages there is often only one topic, so this may frequently
  degenerate to S2. That is a finding, not a failure, and we will report the actual split-rate
  distribution. Using a *per-document* percentile rather than a global threshold is what makes
  this work across languages with different embedding-distance scales.
- **Cost:** an extra embedding pass at ingest. Zero query-time cost.

### S5 · `parent_child` — small-to-big

Index small children (~1–2 sentences). Return the full parent passage.

- **Hypothesis:** the cleanest decoupling of retrieval precision from generation context, and the
  strategy most likely to win outright.
- **Cost:** parent lookup and dedup at query time (multiple children of one parent must collapse
  to a single parent, or the context window fills with duplicates). Counted in `t_core`.

### S6 · `late_chunking` — document-conditioned chunk embeddings

Encode the whole passage in one forward pass with a long-context encoder, then **mean-pool the
token embeddings per chunk span**. Each chunk vector therefore carries information from the entire
document, unlike independently-embedded chunks.

- **Hypothesis:** resolves the classic failure where a chunk says "he was born there" and, embedded
  in isolation, matches nothing — because the pooled vector still contains the document's subject.
- **Note:** the technique from Jina AI's late-chunking work. Requires an encoder whose token
  embeddings we can access, which our ONNX setup gives us.
- **Cost:** ingest-side only. **Zero query-time cost.** Best cost/benefit ratio in the list.

### S7 · `contextual_prefix` — Anthropic-style contextual retrieval

At ingest, generate a one-sentence situating context per chunk with a cheap LLM, prepend it to the
chunk text *before embedding*, but return the original chunk text to the reader.

- **Hypothesis:** substantial recall gain, per the published Contextual Retrieval results.
- **Note:** the clearest example of "embedded text ≠ returned text". Batched and cached offline.
- **Cost:** one cheap LLM call per chunk at ingest — the most expensive strategy here, so it is
  applied to a sampled subset first to measure whether the gain justifies the bill before we run
  it corpus-wide. Zero query-time cost.

### S8 · `proposition` — atomic fact decomposition

Decompose each passage into standalone, self-contained factual propositions (dense-X style), each
independently intelligible with no pronouns or dangling references.

- **Hypothesis:** propositions are the highest-precision retrieval unit and give **sentence-level
  citations**, which directly strengthens the groundedness guardrail — we can attribute each claim
  in the answer to a specific proposition.
- **Cost:** LLM pass at ingest; index inflation. Sampled-subset evaluation first, as with S7.

### S9 · `doc2query_multivector` — generated-question indexing

For each passage, generate 3–5 hypothetical questions it answers. Index those questions as
additional vectors **pointing at the same chunk**.

- **Hypothesis:** the single biggest win for this workload. The task is question-answering, and
  matching a user's question against a *question* is far easier than matching it against a
  declarative passage — it closes the query/document asymmetry gap directly. This is doc2query /
  HyDE run in the correct direction: offline, on the corpus, instead of online, on the query.
- **Bonus:** the dataset already contains a real question per passage set (`query` / `Eng_Query`).
  So we get one gold generated-question **for free, with no LLM call**, and can measure how much
  of the gain comes from the free one versus the generated ones. That is a cheap, high-signal
  experiment.
- **Cost:** ingest LLM pass (partly free, see above); 3–5× vectors. Zero query-time cost.

### S10 · `crosslingual_twin` — parallel twin indexing

Index the Indic chunk and its aligned English twin as linked units in the same collection. A query
in any language can match either; the answer is returned in the query's language via the twin
pointer.

- **Hypothesis:** recovers recall lost to bad machine translation. If a Hindi passage was
  mistranslated, the query will not match it — but the English original is intact and will match a
  cross-lingually embedded query.
- **Why it is ours:** this strategy is only available because MSMARCO-XI is a parallel corpus. It
  is the strategy most specific to the provided dataset, and the one a generic RAG template cannot
  produce.
- **Cost:** 2× vectors. Query-time twin resolution ~1 ms.

### S11 · `script_aware_sizing` — per-language token budgets

Chunk sizes set in **tokens per language**, calibrated by measured characters-per-token, not in
characters or a single global token count.

- **Hypothesis:** a 300-character chunk holds very different amounts of information in Devanagari,
  Tamil and Latin script, and multilingual tokenisers have markedly different fertility across
  Indic scripts. Uniform sizing silently gives some languages far less context than others.
- **Purpose:** this is the multilingual-correctness strategy. Most pipelines get this wrong and
  never notice, because they only test in English.
- **Cost:** a calibration pass to measure per-language fertility. Then free.

### S12 · `hybrid_multi_index` — the ensemble

The champion configuration: run several strategies simultaneously as **named vectors in one Qdrant
collection**, retrieve from each, and fuse with Reciprocal Rank Fusion.

- **Hypothesis:** strategies fail on different queries, so fusion beats any single strategy.
  Expected shape of the winner: `parent_child` + `doc2query` + `crosslingual_twin`, fused with
  BM25 lexical, with `late_chunking` embeddings underneath.
- **Cost:** the honest problem — each additional retrieval arm costs milliseconds. The ablation
  must report **quality *and* latency per arm** so the final choice is a defensible point on the
  Pareto frontier, not simply "more is better". Expect the shipped ensemble to be 2–3 arms, not 6.

---

## Overlap handling, explicitly

The brief names overlap. Each strategy handles it differently, and that variety is the point:

| Strategy | Overlap mechanism |
|---|---|
| S1 | Fixed 64-token stride — blind, may duplicate or split arbitrarily |
| S2 | None — passages are disjoint by construction |
| S3 | Neighbour windows — overlap created at *read* time, not index time |
| S4 | None at boundaries; boundaries are chosen to be low-information points |
| S5 | Hierarchical containment — children overlap only through their shared parent |
| S6 | No text overlap, but *information* overlap via document-conditioned pooling |
| S12 | Deliberate cross-strategy redundancy, deduplicated by RRF and parent collapse |

Duplicate suppression at assembly time is mandatory: with S5 + S12 the top-k will contain multiple
children of one parent, and naively concatenating them wastes the context budget on repeated text.
Assembly deduplicates by `parent_id`, merges adjacent `char_span`s, and fills to a token budget in
fused-rank order.

---

## The ablation plan

Every strategy is benchmarked on the identical query set with the identical retrieval stack, so
the only variable is chunking. Table to be published in `bench/results/chunking_ablation.csv` and
reproduced in the README.

| id | strategy | chunks | Recall@5 | Recall@20 | nDCG@10 | MRR | ingest min | p50 retr ms | p100 retr ms |
|---|---|---|---|---|---|---|---|---|---|
| S1 | fixed_512_64 | | | | | | | | |
| S2 | passage_native | | | | | | | | |
| S3 | sentence_window | | | | | | | | |
| S4 | semantic_drift | | | | | | | | |
| S5 | parent_child | | | | | | | | |
| S6 | late_chunking | | | | | | | | |
| S7 | contextual_prefix | | | | | | | | |
| S8 | proposition | | | | | | | | |
| S9 | doc2query_multivector | | | | | | | | |
| S10 | crosslingual_twin | | | | | | | | |
| S11 | script_aware_sizing | | | | | | | | |
| S12 | hybrid_multi_index | | | | | | | | |

Plus a **per-language breakdown** — a strategy that wins on English and loses on Tamil is a
finding worth reporting, and given that the corpus is machine-translated it is a likely one.

### Commitments about the ablation

- **Latency is a column, not a footnote.** A strategy that wins on nDCG and costs 90 ms loses.
- **We report losers.** If S2 beats S7 and S8, that goes in the table and in the video. Negative
  results measured honestly are stronger evidence of real work than a table where everything
  conveniently improves.
- **S7, S8 and S9 are cost-gated.** They need LLM passes at ingest. They get evaluated on a T0
  subset first; they only reach the full index if the measured gain justifies the spend.
- **The champion is chosen by data**, and the selection rule is written down *before* the numbers
  land: highest nDCG@10 among configurations whose p100 retrieval latency leaves ≥ 60 ms of
  headroom in the `t_core` budget.

---

## Minimum viable scope

If the timeline compresses, the priority order is:

1. **S2, S5, S9** — baseline, the likely winner, and the biggest expected gain. Three strategies
   with genuinely different mechanisms.
2. **S3, S10** — cheap, and S10 is the dataset-specific differentiator.
3. **S6** — best cost/benefit of the sophisticated options, zero query cost.
4. **S1** — the control. Cheap to add and makes the table legible.
5. **S11** — correctness rather than headline score.
6. **S4, S7, S8, S12** — highest cost, most likely to be cut.

Six strategies with a published ablation table is comfortably "vast", and beats twelve
half-implemented ones with no numbers.
