"""Tests for FastAPI realtime voice server endpoints."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from kbee import realtime_server


class _DummyResponse:
    def __init__(self) -> None:
        self.source_nodes = [
            SimpleNamespace(
                metadata={"file_name": "faq.md"},
                score=0.91,
                text="KB answer preview text",
            )
        ]

    def __str__(self) -> str:
        return "mocked kb answer"


def test_health() -> None:
    """Health endpoint should return ok."""
    client = TestClient(realtime_server.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_rag_returns_answer_and_sources(monkeypatch) -> None:
    """RAG endpoint should return answer text with extracted sources."""

    def mock_get_query_engine():
        return object()

    async def mock_query_knowledge_base(query_engine, question: str):
        assert query_engine is not None
        assert question == "營業時間是什麼？"
        return _DummyResponse()

    monkeypatch.setattr(realtime_server, "get_query_engine", mock_get_query_engine)
    monkeypatch.setattr(
        realtime_server,
        "query_knowledge_base",
        mock_query_knowledge_base,
    )

    client = TestClient(realtime_server.app)
    response = client.post(
        "/api/rag",
        json={"query": "營業時間是什麼？"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "mocked kb answer"
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["file_name"] == "faq.md"


def test_api_realtime_token(monkeypatch) -> None:
    """Realtime token endpoint should forward OpenAI session payload."""

    async def mock_create_realtime_session(**kwargs):
        assert kwargs["api_key"] == "test-openai-key"
        assert kwargs["model"] == "gpt-realtime-test"
        assert kwargs["voice"] == "alloy"
        return {
            "id": "sess_123",
            "model": kwargs["model"],
            "client_secret": {"value": "ek_test"},
        }

    monkeypatch.setattr(realtime_server.settings, "openai_api_key", "test-openai-key")
    monkeypatch.setattr(realtime_server.settings, "realtime_model", "gpt-realtime-test")
    monkeypatch.setattr(
        realtime_server,
        "_create_realtime_session",
        mock_create_realtime_session,
    )

    client = TestClient(realtime_server.app)
    response = client.post("/api/realtime-token", json={"voice": "alloy"})

    assert response.status_code == 200
    assert response.json()["client_secret"]["value"] == "ek_test"
    assert response.json()["model"] == "gpt-realtime-test"
