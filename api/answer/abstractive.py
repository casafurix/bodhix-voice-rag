"""Abstractive fast-path answer — NVIDIA LLM, grounded by prompt constraint.
See docs/13-build-status.md.

Citation design: the model is asked to end its answer with a `SOURCES: 1,3`
line referencing the *numbered* context blocks it was given. We map those
numbers back to `chunk_id`s by list index — never by trusting a model-
emitted chunk_id string — so a hallucinated citation is structurally
impossible. The parser tolerates real-world formatting drift (e.g.
"SOURCES: [1][2]" or "1 and 3", observed from the live model) by extracting
any digits on that line rather than requiring an exact comma-separated
match. A missing SOURCES line, or one with no usable digits, falls back to
citing every supplied block — always a true statement, and keeps
guard_out's citation-integrity check meaningful even when the model
doesn't follow the format perfectly.

LATENCY NOTE: `nvidia/llama-3.3-nemotron-super-49b-v1.5` is a reasoning
model — it emits chain-of-thought into a separate `reasoning_content`
field before `content`, and observed live calls take 8-25s wall-clock even
for a short RAG answer (see api/llm/nvidia_client.py's AGENERATE_TIMEOUT_S
and docs/13-build-status.md). "detailed thinking off" is NVIDIA's
documented convention for this model family to shorten (not eliminate)
that reasoning phase — prepended to the system prompt below. Without it,
and with too small a max_tokens budget, the model can spend its entire
budget on reasoning and return empty `content` (finish_reason="length") —
confirmed live during this build; api/llm/nvidia_client.py treats that as
a failure so the harness falls back to extractive rather than surfacing
an empty answer.
"""

from __future__ import annotations

import re

import regex
from pydantic import BaseModel

from api.harness.deadline import Deadline
from api.llm.nvidia_client import agenerate_answer
from api.retrieval.assemble import AssembledChunk

_SOURCES_LINE_RE = regex.compile(r"SOURCES:\s*(.+)\s*$", regex.IGNORECASE | regex.MULTILINE)

_SYSTEM_PROMPT = (
    "detailed thinking off\n\n"
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the numbered context passages below. If the passages do not "
    "contain the answer, say so plainly instead of guessing. Answer in the same "
    "language as the question. Be concise (2-4 sentences). After your answer, on "
    "its own final line, write \"SOURCES:\" followed by a comma-separated list of "
    "the passage numbers you actually used (e.g. \"SOURCES: 1,3\"). If you used "
    "none, write \"SOURCES: none\"."
)


class AbstractiveAnswer(BaseModel):
    text: str
    cited_chunk_ids: list[str]


def _build_prompt(query: str, blocks: list[AssembledChunk]) -> list[dict]:
    numbered = "\n".join(f"[{i + 1}] {b.text}" for i, b in enumerate(blocks))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{numbered}\n\nAnswer:",
        },
    ]


def _parse_sources(raw: str, blocks: list[AssembledChunk]) -> tuple[str, list[str]]:
    match = _SOURCES_LINE_RE.search(raw)
    text = _SOURCES_LINE_RE.sub("", raw).strip() if match else raw.strip()

    cited: list[str] = []
    if match and "none" not in match.group(1).strip().lower():
        indices = {int(n) for n in re.findall(r"\d+", match.group(1))}
        cited = [blocks[n - 1].chunk_id for n in sorted(indices) if 1 <= n <= len(blocks)]

    if not cited:
        cited = [b.chunk_id for b in blocks]

    return text, cited


async def generate_answer(
    query: str, blocks: list[AssembledChunk], deadline: Deadline
) -> AbstractiveAnswer:
    raw = await agenerate_answer(_build_prompt(query, blocks), deadline)
    text, cited_chunk_ids = _parse_sources(raw, blocks)
    return AbstractiveAnswer(text=text, cited_chunk_ids=cited_chunk_ids)
