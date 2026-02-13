# AGENTS.md — KBee

## 專案概覽
KBee 是一個輕量級 AI 客服 SaaS — 店家上傳知識庫文件，自動生成可嵌入的 AI 客服 chat widget。
MVP 階段：單租戶、本地 Chroma vector store、Chainlit chat UI。

## 技術棧
- **語言：** Python 3.11+
- **RAG 框架：** LlamaIndex
- **Vector Store：** ChromaDB（本地持久化）
- **LLM：** OpenAI GPT-4o-mini（via LlamaIndex）
- **Embedding：** OpenAI text-embedding-3-small
- **Chat UI：** Chainlit
- **套件管理：** uv + pyproject.toml

## 專案結構
```
kbee/
├── AGENTS.md
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── src/
│   └── kbee/
│       ├── __init__.py
│       ├── config.py          # Settings via pydantic-settings
│       ├── ingest.py          # 文件 ingestion pipeline
│       ├── query.py           # RAG query engine
│       └── app.py             # Chainlit chat UI entry point
├── data/
│   └── .gitkeep              # 上傳文件放這裡
├── storage/
│   └── .gitkeep              # Chroma persistent storage
└── tests/
    ├── __init__.py
    └── test_ingest.py
```

## Coding 規範
- **Style Guide：** Google Python Style Guide
- **Docstring：** Google 格式（Args, Returns, Raises）
- **命名：** snake_case（函數/變數）、PascalCase（類別）
- **行長：** 88（black default）
- **Import 順序：** 標準庫 → 第三方 → 本地（isort）
- **Type hints：** 所有 public functions 必須有
- **Formatter：** ruff format
- **Linter：** ruff check

## 建構與測試
```bash
# 安裝依賴
uv sync

# 跑 ingestion
uv run python -m kbee.ingest --dir data/

# 啟動 chat UI
uv run chainlit run src/kbee/app.py

# 跑測試
uv run pytest tests/ -v

# Lint + Format
uv run ruff check src/
uv run ruff format src/
```

## 開發注意事項
- 不要 commit .env（已在 .gitignore）
- storage/ 目錄是 Chroma 持久化資料，不 commit 內容
- data/ 目錄放測試用文件，不 commit 實際客戶資料
- MVP 不做多租戶、不做 LINE 串接、不做付款
