"""FastAPI app — POST /ask (text), POST /listen (voice), GET /healthz.
See docs/01-architecture.md, "Interfaces".

/listen is a batch HTTP multipart endpoint (upload a complete audio clip),
not a WebSocket — voice input is speak-then-process, not live streaming.
See docs/13-build-status.md and api/harness/pipeline_voice.py.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.harness.pipeline import run_ask
from api.harness.pipeline_voice import run_ask_voice
from api.schemas import AskRequest, AskResponse

app = FastAPI(title="BodhiX Voice RAG API")

# The frontend (Vercel/Netlify) and this API (Render) deploy to different
# origins — without this, every browser request from the deployed frontend
# is blocked before it even reaches a route. No cookies/auth headers are
# used anywhere in this API, so allow_credentials=False + a configurable
# origin list (settings.CORS_ORIGINS, "*" by default) carries no
# credential-leak risk. See api/config.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict:
    # TODO: readiness should also check index-loaded + models-warm once
    # ingest/build_index.py produces a real artefact — see docs/10-deployment.md.
    return {"status": "ok", "corpus_tier": settings.corpus_tier}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    return await run_ask(request)


@app.post("/listen", response_model=AskResponse)
async def listen(
    audio: UploadFile,
    lang_hint: str | None = Form(None),
    budget_ms: float | None = Form(None),
    answer_mode: Literal["extractive", "abstractive"] = Form("abstractive"),
) -> AskResponse:
    audio_bytes = await audio.read()
    return await run_ask_voice(
        audio=audio_bytes,
        lang_hint=lang_hint,
        budget_ms=budget_ms or settings.voice_default_budget_ms,
        answer_mode=answer_mode,
    )
