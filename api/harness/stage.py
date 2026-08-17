"""Stage protocol + timing decorator. See docs/06-harness.md.

Every stage is a typed, timed unit. `optional` stages are the ones the
degradation ladder in pipeline.py is allowed to skip under budget pressure.
"""

from __future__ import annotations

import time
from typing import Protocol, TypeVar

from api.harness.context import Context

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Stage(Protocol[InputT, OutputT]):
    name: str
    optional: bool

    async def run(self, ctx: Context, inp: InputT) -> OutputT: ...


class StageShortCircuit(Exception):
    """Raised by a stage that wants to terminate the request early with a
    refusal (guard_in, coverage_gate, guard_out). Caught once, at the top
    of pipeline.py.
    """

    def __init__(self, refusal_code: str, detail: str = ""):
        self.refusal_code = refusal_code
        self.detail = detail
        super().__init__(f"{refusal_code}: {detail}")


async def timed(ctx: Context, stage_name: str, coro):
    """Wrap a stage call, recording elapsed ms into ctx.timings_ms regardless
    of success or failure, so a raised StageShortCircuit still leaves a
    complete trace.
    """
    start = time.perf_counter()
    try:
        return await coro
    finally:
        ctx.record_timing(stage_name, (time.perf_counter() - start) * 1000.0)
