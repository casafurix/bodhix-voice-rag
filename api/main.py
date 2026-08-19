"""FastAPI app — POST /ask (text, for benchmarking/debugging), GET /healthz.
See docs/01-architecture.md, "Interfaces".

WS /listen (the voice path) is added once api/stt/sarvam.py + the frontend
are both ready to be wired together — see the todo list / roadmap.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.config import settings
from api.harness.pipeline import run_ask
from api.schemas import AskRequest, AskResponse

app = FastAPI(title="BodhiX Voice RAG API")


@app.get("/healthz")
async def healthz() -> dict:
    # TODO: readiness should also check index-loaded + models-warm once
    # ingest/build_index.py produces a real artefact — see docs/10-deployment.md.
    return {"status": "ok", "corpus_tier": settings.corpus_tier}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    return await run_ask(request)
