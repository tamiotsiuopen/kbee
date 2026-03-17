# KBee

Lightweight AI customer service powered by RAG. Supports both text chat (Chainlit) and realtime voice conversations (WebRTC). Upload your knowledge base documents and KBee turns them into a bilingual AI assistant that answers in English or Traditional Chinese.

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

Sample FAQ data is included in `data/faq/` (English iGaming + Chinese e-commerce). To ingest:

```bash
uv run python -m kbee.ingest --dir data/sporty_stage_ng --clear
```

You can also add your own files (PDF, TXT, DOCX, MD, HTML) to `data/` before ingesting.

### 4. Start the Realtime Voice Server

```bash
uv run uvicorn kbee.realtime_server:app --host 0.0.0.0 --port 8787 --reload
```

Open [http://localhost:8787](http://localhost:8787) (must be `localhost`, not `127.0.0.1`) and click **Start Session**.

### 5. Start the Text Chat UI (optional)

```bash
uv run chainlit run src/kbee/app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Features

- **Bilingual support** -- defaults to English, auto-switches to Traditional Chinese based on user language
- **RAG-grounded answers** -- all questions are answered from the knowledge base, never hallucinated
- **Graceful fallback** -- when the knowledge base has no answer, offers to transfer to a human specialist
- **Low-latency voice** -- retriever-only RAG (no redundant LLM call) for fast voice responses
- **VAD tuning** -- 800ms silence threshold for natural conversation flow

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
│       └── realtime_voice_poc.html
├── data/
│   └── faq/
│       ├── en/                # English FAQ (iGaming, 20 files)
│       └── zh_tw/             # Chinese FAQ (e-commerce, 20 files)
├── storage/                   # ChromaDB persistent storage (git-ignored)
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
- **Vector Store:** ChromaDB 1.5 (persistent, local)
- **LLM (text chat):** OpenAI GPT-5.4
- **Realtime Voice:** OpenAI gpt-realtime (WebRTC + server VAD)
- **Embedding:** OpenAI text-embedding-3-small
- **Text Chat UI:** Chainlit
- **Voice Server:** FastAPI + uvicorn

