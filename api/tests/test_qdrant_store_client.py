from api.config import settings
from api.retrieval import qdrant_store


def test_get_client_memory_mode(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_local_path", ":memory:")
    qdrant_store._reset_client_for_tests()
    client = qdrant_store.get_client()
    try:
        assert client is qdrant_store.get_client()  # singleton
    finally:
        qdrant_store._reset_client_for_tests()


def test_get_client_disk_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_local_path", str(tmp_path / "qdrant"))
    qdrant_store._reset_client_for_tests()
    client = qdrant_store.get_client()
    try:
        assert client is qdrant_store.get_client()
    finally:
        qdrant_store._reset_client_for_tests()
