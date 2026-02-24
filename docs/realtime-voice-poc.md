# KBee Realtime Voice POC

此 POC 使用 OpenAI Realtime Voice (WebRTC) + KBee RAG，實作低延遲語音客服互動。

## 1) 安裝依賴

```bash
uv sync --extra dev
```

## 2) 設定環境變數

```bash
cp .env.example .env
```

請至少確認以下欄位：

- `OPENAI_API_KEY`
- `REALTIME_MODEL` (預設 `gpt-4o-realtime-preview-2024-12-17`)
- `CHROMA_PERSIST_DIR` (預設 `./storage`)
- `DATA_DIR` (預設 `./data`)

## 3) 匯入知識庫文件

將文件放進 `data/` 後執行：

```bash
uv run python -m kbee.ingest --dir data/
```

若要清空舊資料重建：

```bash
uv run python -m kbee.ingest --dir data/ --clear
```

## 4) 啟動 Realtime Voice POC server

```bash
uv run python -m kbee.realtime_server
```

如需手動指定 host/port（例如開發時 reload）：

```bash
uv run uvicorn kbee.realtime_server:app --host 0.0.0.0 --port 8787 --reload
```

## 5) 開啟 POC 頁面

瀏覽器開啟：

- `http://localhost:8787/`

按 `Start Session` 後允許麥克風，開始語音對話。

## 真人互動模擬腳本（Latency Check）

1. 「你好，請先自我介紹，並用一句話說你可以幫我做什麼。」
2. 「請問你們的退貨政策是什麼？」
3. 「如果商品有瑕疵，我幾天內可以申請退款？」
4. 「Can you summarize the shipping policy in English?」
5. 「請根據知識庫告訴我會員方案差異，精簡三點。」

觀察指標：

- 語音回覆延遲是否可接受（從你停頓到 AI 開口）。
- 是否有正確呼叫 RAG（可在頁面 log 看到 `Tool call rag_query`）。
- 知識題是否根據知識庫作答，且中英語切換正常。
