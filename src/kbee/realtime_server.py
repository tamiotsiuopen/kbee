"""FastAPI server for KBee realtime voice POC."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kbee.config import settings
from kbee.query import get_query_engine, query_knowledge_base

REALTIME_SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
POC_HTML_PATH = Path(__file__).with_name("static") / "realtime_voice_poc.html"


class RealtimeTokenRequest(BaseModel):
    """Request body for minting a realtime ephemeral token."""

    voice: str = Field(default="verse", min_length=1)


class RagRequest(BaseModel):
    """Request body for querying knowledge base."""

    query: str = Field(min_length=1)


class RagResponse(BaseModel):
    """Response body for RAG query results."""

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


def _extract_sources(response: Any) -> list[dict[str, Any]]:
    """Extract source metadata from a LlamaIndex response object."""
    source_nodes = getattr(response, "source_nodes", None)
    if not source_nodes:
        return []

    results: list[dict[str, Any]] = []
    for node in source_nodes:
        metadata = getattr(node, "metadata", {}) or {}
        results.append(
            {
                "file_name": metadata.get("file_name", "Unknown"),
                "score": getattr(node, "score", None),
                "text_preview": getattr(node, "text", "")[:160],
            }
        )
    return results


async def _create_realtime_session(
    *,
    api_key: str,
    model: str,
    voice: str,
    instructions: str,
) -> dict[str, Any]:
    """Create a realtime session via OpenAI Realtime API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "voice": voice,
        "modalities": ["audio", "text"],
        "instructions": instructions,
        "tool_choice": "auto",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            REALTIME_SESSION_URL,
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to create realtime session: {response.text}",
        )

    return response.json()


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(title="KBee Realtime Voice POC")

    @app.get("/")
    async def index() -> FileResponse:
        if not POC_HTML_PATH.exists():
            raise HTTPException(status_code=500, detail="POC HTML not found")
        return FileResponse(POC_HTML_PATH)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/realtime-token")
    async def realtime_token(payload: RealtimeTokenRequest) -> dict[str, Any]:
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required")

        instructions = (
            "你是 KBee 客服語音助理。預設使用繁體中文回覆；"
            "若使用者以英文提問，可用英文回覆。"
            "知識查詢優先透過 rag_query 工具。"
        )
        session = await _create_realtime_session(
            api_key=settings.openai_api_key,
            model=settings.realtime_model,
            voice=payload.voice,
            instructions=instructions,
        )
        return session

    @app.post("/api/rag", response_model=RagResponse)
    async def rag(payload: RagRequest) -> RagResponse:
        try:
            query_engine = get_query_engine()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response = await query_knowledge_base(query_engine, payload.query)
        return RagResponse(answer=str(response), sources=_extract_sources(response))

    return app


app = create_app()


def _resolve_host_port(base_url: str) -> tuple[str, int]:
    """Resolve host and port from a base URL with safe defaults."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"

    if parsed.port is not None:
        return host, parsed.port
    if parsed.scheme == "https":
        return host, 443
    return host, 8787


def main() -> None:
    """Run the realtime POC server with uvicorn."""
    import uvicorn

    host, port = _resolve_host_port(settings.base_url)
    uvicorn.run(
        "kbee.realtime_server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
