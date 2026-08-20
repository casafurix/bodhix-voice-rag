# BodhiX Voice RAG — a plain Docker web service, host-agnostic (Render,
# Fly.io, Cloud Run, or local `docker run`). Listens on $PORT if the host
# sets one (Render does), otherwise 7860. See README.md's "Deploying"
# section for the one-time setup steps.
#
# Startup populates the index from the committed parquet cache
# (ingest/embeddings_cache/) rather than re-embedding — see
# ingest/load_cached_embeddings.py. That's ~1 minute, zero NVIDIA/embedding
# API calls, and is what makes a cold start on a free/sleeping tier
# tolerable instead of a 15+ minute re-ingest.

FROM python:3.12-slim

# HF Spaces containers run as a non-root user by convention. Everything
# from `uv sync` onward runs AS that user (not just COPY --chown) — uv
# sync run as root here would leave .venv/~/.cache/uv root-owned, and the
# appuser-run CMD at the bottom would then fail to write to either.
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /home/appuser/app
RUN chown appuser:appuser /home/appuser/app
USER appuser

# Dependencies first, for layer caching.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Source + the pre-built embeddings cache (committed to git, see
# ingest/export_embeddings.py) — everything the app needs to serve
# requests without calling an embedding API at startup.
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser ingest/ ./ingest/

# Best-effort: bake the local fastembed ONNX model into the image so the
# first real request doesn't pay a cold-start download. Harmless if it
# fails during a network-restricted build — it just downloads on first use.
RUN uv run python -c "from api.retrieval.embed import embed_query; embed_query('warmup')" || true

# Disk-backed, but inside the container's own ephemeral writable layer —
# NOT a persistent volume. No :memory: here: the CMD below runs the load
# script and uvicorn as two SEPARATE processes (shell `&&`), and Qdrant's
# :memory: mode is per-process — data populated by the first process
# vanishes the instant it exits, leaving uvicorn's own fresh in-memory
# store empty ("Collection chunks not found" — hit this for real while
# testing locally with `docker run`, confirmed the fix below works).
# Disk mode has no such problem: the load script writes real files, the
# second process opens the same path and finds them. Every container
# start still repopulates from the committed parquet cache (~10-60s, zero
# embedding calls) rather than persisting across restarts.
# Real secrets (SARVAM_API_KEY, NVIDIA_API_KEY) are set as the host's
# environment-variable/secrets feature (e.g. Render's Environment tab),
# never baked into the image.
ENV QDRANT_LOCAL_PATH=/home/appuser/app/ingest/data/qdrant_storage \
    BM25_INDEX_PATH=/home/appuser/app/ingest/data/bm25_index \
    API_HOST=0.0.0.0 \
    PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uv run python -m ingest.load_cached_embeddings && uv run uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
