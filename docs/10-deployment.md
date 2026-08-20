# 10 — Deployment

> **Scope note:** this page is the original 3-person-team plan (Fly.io + Vercel, paid, India-region
> tuned) — kept for its design thinking, not followed literally. What's actually deployed is a free
> Hugging Face Spaces Docker container; see the "Deploying" section in [README.md](../README.md) and
> `Dockerfile`. No frontend yet, so the Vercel/Next.js half below doesn't apply.

The brief requires a **live working link**. A judge will click it, probably once, possibly at 11 pm
on the deadline. It has to work cold, on a phone, on mobile data.

## Topology

```
  users ──▶  Vercel edge (Next.js)      ── web/ ──  region: bom1
              │
              └─ WSS + SSE ──▶  Fly.io machine (FastAPI)  ── api/ ── region: bom
                                  │
                                  ├─ Fly volume: Qdrant + BM25 index (mmap'd)
                                  ├─ outbound: Sarvam  (api.sarvam.ai)
                                  ├─ outbound: ElevenLabs (api.in.residency…)
                                  └─ outbound: Groq / Cerebras (rich path only)
```

**Everything in Mumbai.** This is the single largest latency decision in the project. A US-East API
adds 200–250 ms of round trip to every hop and would make the target unreachable regardless of code
quality. Vercel `bom1`, Fly `bom`, Sarvam is India-hosted, ElevenLabs has an India residency
endpoint. See [08-latency.md](08-latency.md).

---

## Why Fly.io for the API

| Requirement | Consequence |
|---|---|
| Multi-GB index in RAM, mmap'd | Needs a long-lived machine with a persistent volume. **Rules out all serverless.** |
| Warm ONNX sessions | Cold start must not be in the request path |
| Mumbai region | Fly has `bom` |
| Colocated vector DB | Needs a real filesystem and a real process, not a function |
| Predictable CPU for P100 | Dedicated-CPU tier, not shared/burstable |

Serverless is disqualified on the first row alone. Vercel Functions, Lambda and Cloud Run all have
cold starts measured in seconds and no good story for a 750 MB memory-resident index.

Alternatives considered: **Railway** (fine, no India region at time of writing), **Render**
(no India region), **a bare VPS in Mumbai** (cheapest and fastest, but manual ops we do not want
during a nine-day sprint), **Modal** (great for GPU ingest, wrong shape for a persistent low-latency
service). Fly is the best fit for the constraint set.

Machine spec: **4 dedicated vCPU / 8 GB**, one instance, `auto_stop_machines = false`. Scale-to-zero
is explicitly disabled — a scaled-to-zero machine means the judge's first click pays a cold start,
which is the worst possible moment for it.

### On single-instance risk

One machine is a single point of failure. Accepted deliberately: a nine-day project with a two-week
lifetime does not need HA, and a second instance would double cost while introducing index-sync
complexity for no marks. Mitigations: health checks, a monitored uptime ping, and a documented
5-minute redeploy path from the committed index artefact.

---

## The index artefact

The build's most important operational detail. The index is **built offline, snapshotted, and shipped
as an immutable artefact** — never built at deploy time.

```
ingest (GPU box / laptop, hours)
   │
   └─▶ Qdrant snapshot + BM25 index + manifest.json
          │
          ├─ uploaded to object storage (R2 / S3)
          │
          └─▶ Fly release: machine boots, pulls artefact to volume,
              loads, warms models, then passes health check
```

`manifest.json` records the git commit, chunker configuration, embedding model revision, corpus tier,
chunk count and build timestamp. Every benchmark number references an artefact id, so any figure in
the submission can be traced to the exact index that produced it ([09-evaluation.md](09-evaluation.md)).

Deploying a new index is a release with a new artefact id, and rollback is pointing at the previous
one. This also means the index is reproducible from a committed script rather than existing only on
someone's laptop — the classic hackathon failure where the demo dies with a dead SSD.

---

## Cold start and warmup

The readiness probe deliberately fails until the machine is genuinely ready:

1. Process up
2. Volume mounted, artefact present, manifest validated
3. Qdrant loaded, collection count matches manifest
4. BM25 index mmap'd
5. All three ONNX sessions created **and** run once on a dummy input — first inference is slow, and
   it will not be a user's
6. 20 synthetic warmup queries through the full DAG
7. `/healthz` → 200

Only then does Fly route traffic. This is why the readiness gate matters more than usual here: a
machine that accepts traffic at step 3 serves a 3-second first request, and on a latency-graded
submission that first request might be the only one a judge makes.

---

## Frontend

Next.js on Vercel, `bom1`. Static shell, so the page paints instantly regardless of API state.

Points that will actually cause problems if ignored:

- **`getUserMedia` requires HTTPS.** Vercel gives us that. Local dev needs `localhost` (exempt) or a
  tunnel — worth knowing before losing an hour to it.
- **iOS Safari.** The most likely device a judge uses, and the most restrictive: audio capture
  requires a user gesture, `AudioContext` starts suspended and must be resumed on tap, and sample
  rates cannot be assumed. We test on a real iPhone, not just Chrome desktop, because "works on my
  machine" here means "fails in the demo".
- **AudioWorklet** for 16 kHz mono PCM, not `MediaRecorder` — we need raw frames for streaming, and
  `MediaRecorder` gives us containerised WebM.
- **Client-side VAD** to detect end-of-speech; this is what makes `t_stt` measurable from end of
  speech ([05](05-speech-to-text.md)).
- **Text input fallback** — always available. If a judge's microphone permissions fail, the demo
  must still work. A voice-only demo with no fallback is one browser dialog away from scoring zero.
- **Graceful degradation** if the WebSocket fails: fall back to a single `POST /ask`.

---

## Secrets and configuration

| Variable | Where | Notes |
|---|---|---|
| `SARVAM_API_KEY` | Fly secret | Server-side only |
| `ELEVENLABS_API_KEY` | Fly secret | Server-side only |
| `GROQ_API_KEY` / `CEREBRAS_API_KEY` | Fly secret | Rich path only |
| `INDEX_ARTEFACT_ID` | Fly env | Pins the index version |
| `CORPUS_TIER` | Fly env | T0–T3 |
| `RICH_PATH_ENABLED` | Fly env | Off for benchmark runs |
| `NEXT_PUBLIC_API_URL` | Vercel env | Public, non-secret |

No API keys reach the browser. The frontend talks only to our API. `.env.example` is committed;
`.env` is gitignored.

**Abuse control**, because the link is public: per-IP rate limits on `/ask` and `/listen`, a max
audio duration (15 s), a max concurrent-session cap, and a hard monthly spend ceiling on the STT
providers. A public voice endpoint with no cap is an invitation to a surprise bill.

---

## Monitoring

- `/healthz` — liveness + readiness
- `/metrics` — Prometheus: per-stage latency histograms, refusal codes by type, degradation rungs,
  provider failovers, cache hit rate
- Structured JSON logs with `trace_id`, shipped off-box so a machine restart does not lose the
  evidence
- External uptime ping every minute from outside Fly, alerting to the team channel — we want to know
  the link is down before a judge tells us

---

## Cost

| Item | Estimate |
|---|---|
| Fly 4 vCPU / 8 GB dedicated, always-on | ~$60–70 / month |
| Fly volume 20 GB | ~$3 / month |
| Vercel | Free tier |
| Sarvam STT | Pay per audio minute; demo volume is trivial |
| Ingest LLM passes (S7/S8/S9) | One-off, cost-gated in [03](03-chunking.md) |
| Rich-path LLM | Per request, small model, capped |
| GPU for ingest | Few hours rented, one-off |

Well within a hackathon budget. The controllable risks are the ingest LLM passes (gated on measured
benefit) and an uncapped public endpoint (capped).

---

## Pre-submission checklist

- [ ] Live URL works from a phone on mobile data, not just laptop wifi
- [ ] Works on iOS Safari and Android Chrome
- [ ] Cold-start path verified: kill the machine, click the link, measure what a judge would see
- [ ] Text fallback works with the microphone denied
- [ ] Rate limits and spend caps active
- [ ] Index artefact pinned and reproducible from a committed script
- [ ] `README` links resolve; repo is public
- [ ] Both API keys valid and not near a quota limit on 22 August
