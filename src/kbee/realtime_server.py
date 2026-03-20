"""FastAPI server for KBee realtime voice POC."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kbee.config import settings
from kbee.query import retrieve_chunks

REALTIME_SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
STATIC_DIR = Path(__file__).with_name("static")
POC_HTML_PATH = STATIC_DIR / "realtime_voice_poc.html"
KB_HTML_PATH = STATIC_DIR / "kb.html"

_FILENAME_RE = re.compile(r"^(\d+)_(.+)\.txt$")


class RealtimeTokenRequest(BaseModel):
    """Request body for minting a realtime ephemeral token."""

    voice: str = Field(default="shimmer", min_length=1)


class RagRequest(BaseModel):
    """Request body for querying knowledge base."""

    query: str = Field(min_length=1)


class RagResponse(BaseModel):
    """Response body for RAG retrieval results."""

    chunks: list[dict[str, Any]] = Field(default_factory=list)



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

    @app.get("/kb")
    async def kb_page() -> FileResponse:
        if not KB_HTML_PATH.exists():
            raise HTTPException(status_code=500, detail="KB HTML not found")
        return FileResponse(KB_HTML_PATH)

    @app.get("/api/kb-articles")
    async def kb_articles() -> list[dict[str, Any]]:
        data_path = settings.data_path
        if not data_path.exists():
            raise HTTPException(status_code=500, detail="Data directory not found")

        articles: list[dict[str, Any]] = []
        for txt_file in sorted(data_path.rglob("*.txt")):
            match = _FILENAME_RE.match(txt_file.name)
            if not match:
                continue
            article_id = int(match.group(1))
            title = match.group(2).replace("_", " ")
            content = txt_file.read_text(encoding="utf-8", errors="replace")
            articles.append({"id": article_id, "title": title, "content": content})

        articles.sort(key=lambda a: a["id"])
        return articles

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/realtime-token")
    async def realtime_token(payload: RealtimeTokenRequest) -> dict[str, Any]:
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required")

        instructions = (
            "Default language: English. "
            "If user speaks Chinese, reply in Traditional Chinese. "
            "If user speaks English, reply in English. "
            "Any other language: reply in English 'Sorry, we currently only support English and Chinese.' "
            "You are a cheerful and upbeat female customer service voice assistant for SportyBet. Speak in a bright, lively, and warm tone with positive energy. "
            "Greetings/thanks/farewells: reply directly. "
            "All other questions: must call rag_query tool first. "
            "If tool result has 'answer' field, repeat it verbatim."
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
            chunks = await retrieve_chunks(payload.query)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return RagResponse(chunks=chunks)

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
