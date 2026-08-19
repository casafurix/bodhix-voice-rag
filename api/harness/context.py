"""Context carried through the DAG — accumulates everything the response
and the trace need. See docs/06-harness.md.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from api.harness.deadline import Deadline


@dataclass
class RetryRecord:
    stage: str
    attempt: int
    reason: str


@dataclass
class Context:
    deadline: Deadline
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timings_ms: dict[str, float] = field(default_factory=dict)
    degradations: list[str] = field(default_factory=list)
    retries: list[RetryRecord] = field(default_factory=list)
    provider_used: dict[str, str] = field(default_factory=dict)

    def record_timing(self, stage_name: str, ms: float) -> None:
        self.timings_ms[stage_name] = ms

    def degrade(self, reason: str) -> None:
        self.degradations.append(reason)
