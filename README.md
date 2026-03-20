# KBee

Lightweight AI customer service powered by RAG. Supports both text chat (Chainlit) and realtime voice conversations (WebRTC). Upload your knowledge base documents and KBee turns them into an AI voice assistant.

## Access

- **Access Code:** `voiceaipoc`

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest documents

```bash
uv run python -m kbee.ingest --dir data/sporty_stage_ng --clear
```

You can also add your own files (PDF, TXT, DOCX, MD, HTML) to `data/` before ingesting.

### 4. Start the Realtime Voice Server

```bash
uv run uvicorn kbee.realtime_server:app --host 0.0.0.0 --port 8787 --reload
```

Open [http://localhost:8787](http://localhost:8787) and click **Start Session**.

### 5. Start the Text Chat UI (optional)

```bash
uv run chainlit run src/kbee/app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Deployment (Zeabur)

The project includes a `Dockerfile` and `zeabur.json` for one-click deployment on Zeabur.

Required environment variable on Zeabur:
- `OPENAI_API_KEY` -- your OpenAI API key

The container will automatically ingest documents and start the voice server on port **8787**.

## Features

- **RAG-grounded answers** -- all questions are answered from the knowledge base, never hallucinated
- **Graceful fallback** -- when the knowledge base has no answer, offers to transfer to a human specialist
- **Low-latency voice** -- retriever-only RAG (no redundant LLM call) for fast voice responses
- **VAD tuning** -- 800ms silence threshold for natural conversation flow
- **Knowledge base viewer** -- browse ingested articles at `/kb`
- **Password gate** -- access code required to enter the voice POC

## Project Structure

```
kbee/
├── src/kbee/
│   ├── config.py              # Settings (env vars)
│   ├── ingest.py              # Document ingestion (IngestionPipeline)
│   ├── query.py               # RAG retriever + full LLM query engine
│   ├── app.py                 # Chainlit text chat UI
│   ├── realtime_server.py     # FastAPI realtime voice server
│   └── static/
│       ├── realtime_voice_poc.html
│       └── kb.html            # Knowledge base viewer
├── data/
│   └── sporty_stage_ng/       # Knowledge base documents
├── storage/                   # ChromaDB persistent storage (git-ignored)
├── Dockerfile
├── zeabur.json
└── tests/
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint & format
uv run ruff check src/
uv run ruff format src/
```

## Tech Stack

- **RAG Framework:** LlamaIndex (IngestionPipeline + VectorIndexRetriever)
- **Vector Store:** ChromaDB (persistent, local)
- **Realtime Voice:** OpenAI gpt-realtime (WebRTC + server VAD)
- **Embedding:** OpenAI text-embedding-3-small
- **Text Chat UI:** Chainlit
- **Voice Server:** FastAPI + uvicorn
- **Deployment:** Docker + Zeabur
