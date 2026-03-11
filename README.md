# 🐝 KBee

Lightweight AI customer service powered by RAG. Upload your knowledge base documents, and KBee turns them into an AI-powered chat assistant.

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

### 3. Add documents

Place your knowledge base files (PDF, TXT, DOCX, MD) in the `data/` directory.

### 4. Ingest documents

```bash
uv run python -m kbee.ingest
```

To clear existing data and re-ingest:

```bash
uv run python -m kbee.ingest --clear
```

### 5. Start the chat UI (text)

```bash
uv run chainlit run src/kbee/app.py
```

Open http://localhost:8000 in your browser and start chatting!

### 6. Start the Realtime Voice POC

```bash
uv run uvicorn kbee.realtime_server:app --host 0.0.0.0 --port 8787 --reload
```

Open http://localhost:8787 (must be `localhost`, not `127.0.0.1`) and click **Start Session** to begin a voice conversation. The AI queries the knowledge base via RAG in real time.

## Project Structure

```
kbee/
├── src/kbee/
│   ├── config.py              # Settings (env vars)
│   ├── ingest.py              # Document ingestion pipeline
│   ├── query.py               # RAG query engine (retriever + full LLM)
│   ├── app.py                 # Chainlit chat UI
│   └── realtime_server.py     # FastAPI realtime voice server
│   └── static/                # Frontend assets
├── data/                      # Knowledge base documents
│   └── faq/
│       ├── zh_tw/             # Chinese FAQ (e-commerce)
│       └── en/                # English FAQ (iGaming)
├── storage/                   # ChromaDB persistent storage
└── tests/                     # Test suite
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

- **RAG Framework:** LlamaIndex
- **Vector Store:** ChromaDB (local)
- **LLM:** OpenAI GPT-5-mini
- **Realtime Voice:** OpenAI gpt-realtime (WebRTC)
- **Embedding:** OpenAI text-embedding-3-small
- **Chat UI:** Chainlit
- **Voice Server:** FastAPI + uvicorn
