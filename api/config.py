"""Central settings. One source of truth, read once at process start.

Every module that needs a path or a key imports `settings` from here rather
than reading `os.environ` directly — keeps the .env.example file honest.
"""

from functools import lru_cache

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

    # api
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def language_list(self) -> list[str]:
        return [lang.strip() for lang in self.languages.split(",") if lang.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
