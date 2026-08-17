"""Coverage gate — the off-topic detector. See docs/07-guardrails.md.

The insight: the retrieval score distribution already tells you whether the
corpus covers the question. No classifier needed, the signal is free — we
already ran the search.

Thresholds below are placeholders (`TAU_*`) and MUST be replaced by the
calibration sweep in bench/run_guardrails.py once we have in-domain /
out-of-domain query sets built (docs/07-guardrails.md, "Coverage gate").

SCALE BUG FOUND AND FIXED DURING FIRST REAL-DATA TEST: this gate receives
`fused[i].score` from retrieval/fuse.py, i.e. **RRF fusion scores**, not
raw cosine similarity. RRF's `1/(k+rank)` with k=60 summed across up to 7
arms (6 dense strategies + 1 sparse) lands in roughly a 0.01-0.12 range
even for a clearly relevant, highly-ranked hit — nothing like the 0-1
cosine-similarity range the first version of these constants assumed
(0.55/0.45). That mismatch refused a genuinely on-topic, correctly
indexed English query ("what is a corporation", literally present in the
ingested corpus) with top1=0.033 < the old TAU_ABSOLUTE=0.55. Thresholds
below are rescaled to RRF's actual numeric range; they are still
placeholders pending the real calibration sweep, but they no longer
reject on-topic queries by construction.
"""

from __future__ import annotations

from statistics import mean, stdev

from pydantic import BaseModel

from api.harness.stage import StageShortCircuit

# TODO(calibration): replace with values from the ROC sweep in
# bench/run_guardrails.py — see docs/07-guardrails.md. Scaled for RRF
# fusion scores (k=60, up to 7 arms) — see module docstring.
TAU_ABSOLUTE = 0.015
TAU_MEAN = 0.008
TAU_MARGIN = 0.003
TAU_SPREAD = 0.003


class CoverageStats(BaseModel):
    top1: float
    mean5: float
    margin: float
    spread: float


def coverage_verdict(scores: list[float]) -> CoverageStats:
    """Raises StageShortCircuit(OUT_OF_SCOPE | LOW_CONFIDENCE) or returns the
    stats to attach to the response trace on PROCEED.
    """
    if not scores:
        raise StageShortCircuit("OUT_OF_SCOPE", "empty candidate set")

    top1 = scores[0]
    mean5 = mean(scores[: min(5, len(scores))])
    margin = scores[0] - scores[min(9, len(scores) - 1)]
    spread = stdev(scores[: min(20, len(scores))]) if len(scores) > 1 else 0.0

    stats = CoverageStats(top1=top1, mean5=mean5, margin=margin, spread=spread)

    if top1 < TAU_ABSOLUTE:
        raise StageShortCircuit("OUT_OF_SCOPE", f"top1={top1:.3f} < {TAU_ABSOLUTE}")
    if mean5 < TAU_MEAN:
        raise StageShortCircuit("OUT_OF_SCOPE", f"mean5={mean5:.3f} < {TAU_MEAN}")
    if margin < TAU_MARGIN and spread < TAU_SPREAD:
        raise StageShortCircuit("LOW_CONFIDENCE", "flat score distribution")

    return stats
