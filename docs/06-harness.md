# 06 — The harness

> The brief: *"Your model/pipeline should be run inside a proper harness — structured orchestration
> around the model (tool calls, retries, structured input/output handling, error recovery) rather
> than a single raw prompt-in, text-out call."*

## What a harness means at a 200 ms budget

The word "harness" often gets read as "agent framework". That would be the wrong choice here, and
the reason is arithmetic: an agent loop costs one LLM round trip per turn, 200 ms+ each. A
two-turn ReAct loop has spent the entire budget before it retrieves anything. **An agentic loop is
structurally incompatible with this brief's own latency requirement.**

So the harness is an **explicit typed stage DAG with deadline propagation**. It delivers everything
the brief asks for — tool dispatch, retries, structured I/O, error recovery — while being
deterministic, bounded and fast. The interesting engineering is not in dynamic planning; it is in
**budget-aware graceful degradation**: the pipeline knows how much time it has left and chooses a
cheaper strategy rather than blowing the deadline.

That property is the thing worth demonstrating on camera.

---

## Core abstractions

### Deadline

Every request carries a wall-clock deadline from the moment it enters the harness. Stages ask how
much is left; they do not assume.

```python
@dataclass(frozen=True)
class Deadline:
    started_at: float          # perf_counter()
    budget_ms: float

    @property
    def elapsed_ms(self) -> float: ...
    @property
    def remaining_ms(self) -> float: ...
    def affords(self, cost_ms: float) -> bool:
        return self.remaining_ms >= cost_ms
    def child(self, slice_ms: float) -> "Deadline": ...   # sub-deadline for a call
```

`affords()` uses each stage's **measured P95** from a live rolling histogram, not a hardcoded
constant. The harness therefore adapts to the machine it is actually running on — a slower box
sheds optional stages earlier by itself, with no retuning.

### Stage

```python
class Stage(Protocol):
    name: str
    optional: bool                # may the degradation ladder skip this?
    InputT: type[BaseModel]
    OutputT: type[BaseModel]

    async def run(self, ctx: Context, inp: InputT) -> OutputT: ...
```

Pydantic v2 models on both sides of every stage. Structured input and output handling is not a
prompt instruction hopefully obeyed — it is a type boundary that raises on violation.

### Context

Carried through the DAG; accumulates everything the response and the trace need.

```python
class Context(BaseModel):
    trace_id: str
    deadline: Deadline
    timings: dict[str, float]      = {}
    degradations: list[Degradation] = []
    guardrails: GuardrailTrace
    retries: list[RetryRecord]     = []
    provider_used: dict[str, str]  = {}
```

---

## The DAG

```
                        ┌────────────────────┐
                        │ ingress            │  validate, rate-limit, trace_id
                        └─────────┬──────────┘
                                  ▼
                        ┌────────────────────┐
                        │ stt                │  tool call · retry · failover
                        │ [optional: skip if │  (excluded from t_core)
                        │  text input]       │
                        └─────────┬──────────┘
              ══════════ t_core ══╪═══════════════════════════
                                  ▼
                        ┌────────────────────┐
                        │ normalise          │  NFC, script detect, lang id
                        └─────────┬──────────┘
                                  ▼
                        ┌────────────────────┐
                        │ guard_in           │  ── may SHORT-CIRCUIT ──▶ refuse
                        └─────────┬──────────┘
                                  ▼
                        ┌────────────────────┐
                        │ embed              │  local ONNX, no network
                        └─────────┬──────────┘
                                  ▼
                     ┌────────────┴────────────┐
                     ▼                         ▼      concurrent
              ┌─────────────┐          ┌─────────────┐
              │ retrieve_   │          │ retrieve_   │  tool calls
              │ dense       │          │ sparse      │
              └──────┬──────┘          └──────┬──────┘
                     └────────────┬───────────┘
                                  ▼
                        ┌────────────────────┐
                        │ fuse (RRF)         │
                        └─────────┬──────────┘
                                  ▼
                        ┌────────────────────┐
                        │ coverage_gate      │  ── may SHORT-CIRCUIT ──▶ refuse
                        └─────────┬──────────┘         OUT_OF_SCOPE
                                  ▼
                        ┌────────────────────┐
                        │ rerank  [OPTIONAL] │  ◀── first to be shed
                        └─────────┬──────────┘
                                  ▼
                        ┌────────────────────┐
                        │ assemble           │  parent expand, dedup, budget
                        └─────────┬──────────┘
                                  ▼
                        ┌────────────────────┐
                        │ answer_fast        │  extractive, always runs
                        └─────────┬──────────┘
                                  ▼
                        ┌────────────────────┐
                        │ guard_out          │  ── may VETO ──▶ refuse
                        └─────────┬──────────┘         UNGROUNDED_ANSWER
              ══════════ t_core ══╪═══════════════════════════
                                  ▼
                            emit response
                                  │
                        ┌─────────▼──────────┐
                        │ answer_rich        │  async, fire-and-forget,
                        │ [OPTIONAL]         │  re-enters guard_out;
                        └────────────────────┘  dropped silently on failure
```

Three stages can terminate the request early with a refusal. That is the guardrail system having
real authority over the pipeline rather than being an advisory post-processing step.

---

## Tool calls

The brief names tool calls. The harness dispatches to typed tools with schemas, timeouts and
retry policies — the same discipline an agent framework applies, minus the LLM deciding which to
invoke, because at this budget the routing is deterministic and we know it statically.

| Tool | Type | Timeout | Retry | On failure |
|---|---|---|---|---|
| `stt.transcribe` | Network | 4000 ms | 1, jittered | Failover to secondary provider |
| `stt.stream` | Network WS | — | reconnect ×2 | Fall back to non-streaming |
| `embed.query` | Local ONNX | 50 ms | 0 | Fail request (unrecoverable) |
| `vector.search` | Local Qdrant | 60 ms | 1 | Degrade to sparse-only |
| `sparse.search` | Local bm25s | 40 ms | 0 | Degrade to dense-only |
| `rerank.score` | Local ONNX | 90 ms | 0 | Skip, use fused order |
| `nli.entail` | Local ONNX | 50 ms | 0 | Degrade to lexical-overlap check |
| `llm.generate` | Network | 5000 ms | 1 | Drop rich path, keep extractive |

Note that **every unrecoverable-on-failure tool is local** and every network tool has a fallback.
That is not an accident; it is the reason the pipeline can promise a bounded response time at all.

There is one place where an LLM *does* choose a tool, well outside the hot path: a `/diagnose`
endpoint used in the demo to explain why a refusal happened. Genuine LLM tool use, zero latency
cost, useful for the video.

---

## Retries and error recovery

```python
RETRY = RetryPolicy(
    max_attempts=2,
    backoff="exponential_jitter",
    base_ms=50,
    retry_on=(TimeoutError, ConnectionError, HTTP5xx, HTTP429),
    never_retry_on=(HTTP4xx_client, ValidationError, GuardrailRefusal),
    deadline_aware=True,        # no retry that cannot finish inside the budget
)
```

`deadline_aware` is the part that matters. A blind retry policy is *worse* than none under a
latency SLO: it turns one slow request into one very slow request. The harness only retries when
the remaining budget can absorb another attempt, and otherwise degrades immediately.

**Circuit breakers** per external dependency: 3 failures in 30 s trips the breaker, half-open after
60 s. A tripped breaker means we stop paying the timeout cost on a provider we already know is
down.

**Idempotency:** every request carries a `trace_id`; retries reuse it, so duplicate work is
detectable in the trace and cached partial results can be reused.

---

## The degradation ladder

Fired when `deadline.remaining_ms` cannot cover the remaining required stages. Rungs are taken in
order, cheapest-loss first:

| # | Degradation | Saves | Quality cost |
|---|---|---|---|
| 1 | Skip `answer_rich` | 300–700 ms (off critical path anyway) | Answer is extractive, not fluent |
| 2 | Skip `rerank` | ~45 ms | Fused order instead of reranked — measured, reported |
| 3 | Reduce candidates 50 → 20 | ~5 ms | Slightly lower recall |
| 4 | Sparse-only or dense-only | ~6 ms | Lose hybrid robustness |
| 5 | Lower HNSW `ef` 64 → 32 | ~3 ms | Lower ANN recall |
| 6 | Cheap lexical groundedness instead of NLI | ~22 ms | Weaker hallucination check |
| 7 | Refuse with `BUDGET_EXCEEDED` | — | No answer, honestly |

Two commitments about this ladder:

- **Rung 7 exists.** The system will return a refusal rather than silently overrun the deadline. A
  latency SLO that is allowed to be violated is not an SLO.
- **Every degradation is reported** in the response `degradations` array and in the benchmark CSV.
  A P100 of 190 ms achieved by degrading 40 % of queries is a different claim from one achieved with
  the full pipeline, and we publish the degradation rate alongside the latency so nobody has to
  guess which one we did.

---

## Structured output

The extractive path produces structured output by construction — spans and chunk ids, no parsing.

The abstractive path is the one that can misbehave, and gets three layers:

1. **Constrained decoding** — JSON schema enforcement where the provider supports it, so malformed
   output is impossible rather than merely unlikely.
2. **Validate** — Pydantic parse. On failure, one repair attempt with the validation error fed back.
3. **Reject** — second failure drops the rich path entirely. The user keeps the extractive answer
   and never sees a parse error.

```python
class RichAnswer(BaseModel):
    answer: str
    cited_chunk_ids: list[str]        # must be a subset of what we supplied
    confidence: Literal["high","medium","low"]
    insufficient_context: bool        # the model's own abstention signal

    @model_validator(mode="after")
    def citations_must_exist(self): ...
```

`insufficient_context` is worth calling out: the model is given an explicit, cheap way to say "I
cannot answer this from what you gave me". Models are far more willing to abstain when abstention
is a structured field than when it requires composing a refusal in prose against an instruction to
be helpful.

---

## Observability

- **OpenTelemetry span per stage**, with timings, input sizes, cache hits, degradations.
- **`timings_ms` in every response** — the demo UI reads the same numbers the benchmark does, so
  there is no risk of the HUD and the CSV disagreeing.
- **Replayable trace log:** every request writes a JSONL record with the transcript, retrieved
  chunk ids, scores, guardrail verdicts and timings. `bench/replay.py` re-runs any historical
  request against the current code — which is how we tell an actual regression from noise.
- **Rolling per-stage histograms** in memory, feeding `affords()`. The harness measures itself.

---

## Testing

| Level | What |
|---|---|
| Unit | Each stage in isolation with fixture I/O. Includes the e5 prefix test — `query:` / `passage:` — because getting it wrong silently costs recall |
| Contract | Every stage's declared Pydantic types actually match its neighbours' |
| Fault injection | Force STT 500s, provider timeouts, Qdrant unavailability, malformed LLM output. **Assert the degradation ladder fires in the documented order** |
| Budget | Run with artificially tight `budget_ms` (150, 100, 60) and assert the correct rungs are taken and the deadline is never exceeded |
| Golden | Fixed query set, assert retrieved chunk ids and refusal codes are stable across commits |
| Load | Concurrency 1 / 4 / 16, assert P100 holds and no unbounded queueing |

The fault-injection and budget suites are the ones that prove the harness claim. A harness that has
never been tested under failure is a diagram, not a harness — and these tests are also the source
of the failover and degradation footage for the demo video.
