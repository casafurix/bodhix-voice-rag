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

    # retrieval / index
    qdrant_local_path: str = "./ingest/data/qdrant_storage"
    bm25_index_path: str = "./ingest/data/bm25_index"
    corpus_tier: str = "T0"
    index_artefact_id: str = "dev"

    # harness
    default_budget_ms: float = 200.0
    languages: str = "en,hi,ta"

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
