"""Central settings. One source of truth, read once at process start.

Every module that needs a path or a key imports `settings` from here rather
than reading `os.environ` directly — keeps the .env.example file honest.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # STT
    sarvam_api_key: str = ""
    elevenlabs_api_key: str = ""  # stub only, not wired in MVP

    # LLM / embedding (NVIDIA NIM, OpenAI-compatible) — see api/llm/nvidia_client.py
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_embed_model: str = "nvidia/nemotron-3-embed-1b"
    nvidia_llm_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    nvidia_embed_dim: int = 2048  # confirmed live: nvidia/nemotron-3-embed-1b returns 2048-dim
    # 35s: nvidia_llm_model (a reasoning model) alone takes 8-25s live for one
    # abstractive answer, on top of STT + retrieval + guardrails. See
    # api/llm/nvidia_client.py's agenerate_answer docstring.
    voice_default_budget_ms: float = 35000.0

    # retrieval / index
    qdrant_local_path: str = "./ingest/data/qdrant_storage"
    bm25_index_path: str = "./ingest/data/bm25_index"
    corpus_tier: str = "T0"
    index_artefact_id: str = "dev"

    # harness
    default_budget_ms: float = 200.0
    languages: str = "en,hi,bn,ta,mr"

    # Memory-constrained deployment mode (Render free tier, 512MB hard cap):
    # the local MiniLM ONNX model is 224MB on disk and, loaded via
    # onnxruntime, was confirmed live to push the container over its cap on
    # the FIRST real query (exit 137 / 502, see docs/13-build-status.md) --
    # even after quantizing the index and tuning onnxruntime's own session
    # options. ONE flag, not two: MEMORY_CONSTRAINED_DEPLOY=true routes text
    # /ask through the same online NVIDIA embedding voice queries already
    # use, AND stops the coverage gate's own local-model fallback
    # (api/harness/pipeline.py) from loading it anyway -- both derived from
    # this single setting rather than two independently-settable ones,
    # after a live deploy still crashed because only one of an earlier
    # two-flag version actually took effect (root cause never fully
    # confirmed -- this removes the failure mode by construction instead of
    # continuing to chase it). Local dev is unaffected: defaults to False,
    # meaning the exact prior behavior for both.
    memory_constrained_deploy: bool = False
    # Coarser than the MiniLM calibration (bench/run_guardrails_calibration.py,
    # 0.70/0.62) -- NVIDIA nemotron cosine showed weaker in/out-of-domain
    # separation when measured (in-min 0.248 vs out-max 0.270). Best-effort
    # for the free-tier deployment path; not a claim of equal guardrail
    # precision to the calibrated local-model path.
    nvidia_coverage_tau_absolute: float = 0.26
    nvidia_coverage_tau_mean: float = 0.20

    @property
    def embedding_provider(self) -> Literal["local", "nvidia"]:
        return "nvidia" if self.memory_constrained_deploy else "local"

    @property
    def coverage_local_reembed(self) -> bool:
        return not self.memory_constrained_deploy

    # api
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Wildcard by default: the frontend and backend deploy to different
    # origins (Vercel/Netlify vs Render), and the API takes no cookies/auth
    # headers, so a wide-open CORS policy carries no credential-leak risk.
    # Lock this to the real frontend origin(s) via the host's env var once
    # that URL is known, e.g. "https://nova-bodhix.vercel.app".
    cors_origins: str = "*"

    @property
    def language_list(self) -> list[str]:
        return [lang.strip() for lang in self.languages.split(",") if lang.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
