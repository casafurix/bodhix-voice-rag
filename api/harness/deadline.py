"""Deadline propagation — see docs/06-harness.md.

Every request carries a wall-clock budget from the moment it enters the
harness. Stages ask how much is left; they never assume. This is what makes
the degradation ladder in pipeline.py possible: a decision like "skip rerank"
is made from a real number, not a guess.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Deadline:
    budget_ms: float
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    @property
    def remaining_ms(self) -> float:
        return self.budget_ms - self.elapsed_ms

    def affords(self, cost_ms: float) -> bool:
        """Would spending `cost_ms` more still leave the deadline unmissed?"""
        return self.remaining_ms >= cost_ms

    def child(self, slice_ms: float) -> "Deadline":
        """A sub-deadline for a single tool call, capped by what's left."""
        return Deadline(budget_ms=min(slice_ms, max(self.remaining_ms, 0.0)))
