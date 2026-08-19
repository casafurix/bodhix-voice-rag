# 05 — Speech-to-text: Sarvam vs ElevenLabs

> **MVP scope note:** Sarvam adapter (`api/stt/sarvam.py`) is implemented (non-streaming
> `transcribe()` only — no realtime WebSocket streaming, no speculative retrieval).
> **ElevenLabs failover is not wired** — `api/stt/elevenlabs.py` implements the same interface
> as a stub so it can be activated later, but the harness never calls it. Neither adapter has
> been exercised against a real audio clip yet; the pipeline has only been tested via `/ask`'s
> text path so far. Live status: [docs/13-build-status.md](13-build-status.md).

> The brief: *"Use either Sarvam or ElevenLabs for voice-to-text. Pick one."*

## Decision

**Sarvam is the primary.** ElevenLabs is wired as a circuit-breaker failover only.

The graded pipeline uses one provider, as instructed. The second exists so that a live demo link
does not die if a provider has a bad five minutes at the moment a judge clicks it. We consider
shipping a single-provider dependency with no fallback on a submission that is graded partly on a
*live working link* to be the wrong call, and the failover is ~40 lines behind an interface we need
anyway.

---

## Why Sarvam

| Criterion | Sarvam | ElevenLabs |
|---|---|---|
| Indic language coverage | **22 Indian languages**, purpose-built | 90+ languages, general-purpose; Indic is not the design centre |
| Match to this corpus | MSMARCO-XI **is** an Indic dataset from AI4Bharat | Mismatch: strongest where we need it least |
| Hosting locality | India | Multi-region incl. `api.in.residency.elevenlabs.io` |
| Code-mixed Hindi/English | **`codemix` mode** — a first-class feature | Handled implicitly |
| Transliteration | **`translit` mode** — Devanagari → Roman | Not offered |
| Query-time translation | **`translate` mode** — Indic speech → English text in one call | Separate step |
| Number normalisation | Explicit, mode-controlled | Implicit |
| Realtime streaming | WebSocket streaming STT | Realtime STT WebSocket (`scribe_v2_realtime`) |
| Word-level timestamps | No — chunk/phrase level only | **Yes**, plus character-level |
| Per-word confidence | Not exposed | **Yes** — `logprob` per word |
| Diarisation | Batch API only | Yes, on the main endpoint |
| Keyterm biasing | Not offered | **Yes** — `keyterms`, up to 1000 |

Sarvam wins on the axes that matter for *this* dataset in *this* competition, in Goa, judged
by people who will very likely test it in Hindi.

Two Sarvam features are not just nice-to-have, they change the architecture:

**`translate` mode is a free cross-lingual retrieval arm.** A Hindi utterance can be transcribed as
Hindi *and* returned as English text in a single API call. That gives us two query strings for one
round trip, which pairs directly with the `crosslingual_twin` chunking strategy
([03-chunking.md](03-chunking.md) S10) — search the Hindi text against Hindi chunks and the English
text against English twins, then fuse. No extra latency in `t_core`, no translation model to host.

**`codemix` mode matches how people actually speak.** Real Indian-English queries are code-mixed —
"मैनहट्टन project का impact क्या था". A transcript that renders English words in Latin and Hindi words
in Devanagari matches our index better than one that forces everything into one script.

### What we give up, and how we compensate

ElevenLabs' per-word `logprob` would be an excellent input to the input guardrail — low mean
confidence is a clean signal for "the audio was unintelligible, refuse". Sarvam does not expose it.
Compensation: use `language_probability` (returned when `language_code` is `unknown`), transcript
length versus audio duration, and a gibberish heuristic instead. Documented in
[07-guardrails.md](07-guardrails.md) as a known weaker signal rather than papered over.

### What would flip the decision

Written down in advance so the choice stays honest:

- Sarvam Hindi/Tamil WER measured materially worse than ElevenLabs on our own test clips
- Sarvam `t_stt` P95 from Mumbai materially worse than ElevenLabs' India residency endpoint
- Sarvam rate limits too tight to run a 300-query voice benchmark
- Realtime streaming quality unusable in practice on either side

All four are measured in week one, not assumed. See the bake-off below.

---

## The bake-off (day 2–3)

Both providers get measured on the same clips, and the numbers are published either way. If
ElevenLabs wins, we switch and say why — the decision above is a hypothesis with a test attached.

**Test set:** 60 clips — 5 languages × 12 questions, drawn from real `query` fields in the dataset
so the reference transcript is known exactly. Recorded by team members, plus synthesised variants
for consistency. Include deliberately hard cases: code-mixed, background noise, fast speech,
numbers, named entities, and one clip of pure silence.

**Metrics:**

| Metric | Why |
|---|---|
| WER / CER per language | CER matters more for Indic scripts than WER |
| **Retrieval-preserving accuracy** | The real metric: does the transcript retrieve the same top-5 as the gold text? A transcription error that does not change retrieval is harmless; one that does is fatal. This is the number that decides. |
| `t_stt` P50/P95/P100 from Mumbai | Latency, measured from end of speech |
| Streaming partial stability | How much do interim transcripts thrash? Determines whether speculative retrieval is viable |
| Failure behaviour | What happens on silence, noise, 30 s audio, rate limit |

Retrieval-preserving accuracy is the metric worth highlighting: it evaluates STT *by its effect on
the pipeline* rather than in isolation, which is the only thing that matters here.

---

## Integration design

One interface, two adapters, so the provider is a config value:

```python
class SttProvider(Protocol):
    name: str
    async def transcribe(self, audio: bytes, lang_hint: str | None,
                         deadline: Deadline) -> Transcript: ...
    async def stream(self, frames: AsyncIterator[bytes],
                     lang_hint: str | None) -> AsyncIterator[PartialTranscript]: ...

class Transcript(BaseModel):
    text: str
    language: str
    language_confidence: float | None
    translated_text: str | None      # Sarvam translate mode
    provider: str
    duration_s: float
    latency_ms: float
```

### Sarvam adapter

```
POST https://api.sarvam.ai/speech-to-text
header: api-subscription-key
form:   file, model=saaras:v3, mode=transcribe|translate|codemix,
        language_code=unknown (auto-detect) | hi-IN | ta-IN | bn-IN | mr-IN | en-IN
```

- `model=saaras:v3` is the current default; `saaras:v4` is available and covers 22 Indic languages
  plus Global/Indian English. We benchmark both and pin the winner explicitly rather than inheriting
  a default that can change under us.
- `language_code=unknown` gives auto-detection **and** returns `language_probability`, which the
  input guardrail needs. Worth the small accuracy cost of not pinning the language.
- Audio: **16 kHz mono** is the documented sweet spot, which is also what our AudioWorklet
  produces. PCM `pcm_s16le` requires `input_audio_codec` to be set explicitly — an easy bug.
- REST endpoint is for clips under 30 s. Our queries are 2–8 s. Batch API is irrelevant here.

### ElevenLabs adapter (failover)

```
POST https://api.in.residency.elevenlabs.io/v1/speech-to-text
form: file, model_id=scribe_v2, file_format=pcm_s16le_16, language_code=…
```

- India residency endpoint, for latency.
- `file_format=pcm_s16le_16` is explicitly documented as **lower latency** than an encoded
  waveform — we already have raw PCM, so this is free.
- `keyterms` is available for entity biasing if we find named-entity errors dominate; note the 20 %
  surcharge, so it stays off unless the bake-off shows it earns its cost.

---

## Streaming and speculative retrieval

Both providers offer realtime WebSocket STT. This matters for one specific reason: it changes what
`t_stt` even means.

- **Non-streaming:** user stops speaking → encode → upload whole clip → wait → transcript. The
  upload and the entire inference happen *after* speech ends. `t_stt` includes all of it.
- **Streaming:** audio frames flow to the provider *while the user is still talking*. When speech
  ends, only the final commit remains. `t_stt` measured from end-of-speech collapses dramatically.

On top of that, **speculative retrieval**: interim transcripts arrive before the final one. We
begin encoding and retrieval on a stable partial, and if the final transcript differs materially we
discard and redo. When the guess holds — which is common, because the tail of a question rarely
changes its meaning — retrieval is already finished when the final transcript lands, and the user's
perceived `t_core` approaches zero.

Guardrails on speculation, because it can go wrong:

- Only speculate once a partial has been stable for ~150 ms
- Never emit a speculative answer to the user; it only pre-warms retrieval
- Hard cap on speculative attempts per utterance, so a thrashing transcript cannot spin the CPU
- Report speculation hit rate in the benchmark, and report `t_e2e` both with and without it

**Client-side VAD** marks end-of-speech in the browser, so we know when to trigger the final commit
instead of waiting on a provider timeout. This is the single highest-leverage latency optimisation
in the entire voice path, and it is entirely in our control.

---

## Failover

```
Sarvam (primary)
   ├─ 2xx           → done
   ├─ timeout       → 1 retry, jittered backoff, tighter deadline
   ├─ 5xx / 429     → open circuit breaker, route to ElevenLabs
   └─ breaker open  → straight to ElevenLabs, retry Sarvam after cooldown
```

The breaker trips on 3 failures in 30 s and half-opens after 60 s. Every failover is logged and
surfaced in the response `degradations` array, so the demo can show it happening rather than
claiming it. Provider used is recorded per benchmark row, and any run containing failovers is
reported separately — a benchmark silently averaging two providers is meaningless.

---

## Secrets

`SARVAM_API_KEY`, `ELEVENLABS_API_KEY` — server-side only, never shipped to the browser. The
frontend talks to our WebSocket; our server talks to the providers. ElevenLabs' single-use token
flow would permit direct browser→provider streaming, which would cut a hop, but it removes our
ability to run the input guardrail before spending money and is not worth the exposure.
