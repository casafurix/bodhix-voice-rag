"""Coverage gate — the off-topic detector. See docs/07-guardrails.md.

The insight: the retrieval score distribution already tells you whether the
corpus covers the question. No classifier needed, the signal is free — we
already ran the search.

CALIBRATED (bench/run_guardrails_calibration.py, 30 in-domain + 10
out-of-domain queries, T0 corpus): the gate now runs on **raw dense cosine
similarity**, not RRF fusion scores. Two findings drove this:

1. RRF scores are rank-based: an out-of-domain query still produces a full
   top-50 *ranking*, so its fused scores are statistically identical to an
   on-topic query's. The two distributions overlap almost completely on RRF
   scale and no threshold can separate them.
2. Raw cosine separates cleanly:
     in-domain  top1 min = 0.817, mean5 min = 0.713
     out-of-dom top1 max = 0.600, mean5 max = 0.522
   Thresholds below sit at the midpoints of those gaps.

The margin/spread LOW_CONFIDENCE check is DISABLED (thresholds 0.0): on the
cosine scale margin and spread overlap across the two populations (in-domain
margin reaches 0.0 too), so they refuse nothing without refusing real
questions. Revisit with a proper ROC sweep if a secondary signal is needed.
"""

from __future__ import annotations

from statistics import mean, stdev

from pydantic import BaseModel

from api.harness.stage import StageShortCircuit

# Calibrated on raw dense cosine similarity — see module docstring and
# bench/run_guardrails_calibration.py. Midpoint of [0.817, 0.600] / [0.713, 0.522].
TAU_ABSOLUTE = 0.70
TAU_MEAN = 0.62
# Disabled — see module docstring. `margin < 0` is unreachable for cosine
# scores, so the LOW_CONFIDENCE rung cannot fire until recalibrated.
TAU_MARGIN = 0.0
TAU_SPREAD = 0.0


class CoverageStats(BaseModel):
    top1: float
    mean5: float
    margin: float
    spread: float


def coverage_verdict(scores: list[float]) -> CoverageStats:
    """Raises StageShortCircuit(OUT_OF_SCOPE | LOW_CONFIDENCE) or returns the
    stats to attach to the response trace on PROCEED.

    `scores` must be raw dense cosine similarities of the global candidate
    set, highest first (as returned by search_dense_grouped).
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
