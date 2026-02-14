# 對話知識挖掘（Conversation Knowledge Mining）

> 從客服歷史對話中自動提取 FAQ 與應對知識，注入 KBee RAG pipeline。

## 目標

讓 KBee 的客戶不需要手動寫 FAQ，只要提供「過去的客服對話紀錄」，系統就能自動：

1. 把對話按主題分群
2. 從每群提取 FAQ（問題 + 最佳回答 + 規則/例外）
3. 偵測矛盾資訊（例如不同客服給了不同答案）
4. 輸出成 KBee 能 ingest 的 markdown

---

## 架構

```
歷史對話（CSV/JSON）
    │
    ▼
┌──────────────────────────────┐
│  Step 1: 前處理               │
│  - 清洗（去除系統訊息、空行）  │
│  - 每段對話合併成一個 document │
│  - 標記 metadata（日期、客服） │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Step 2: Embedding            │
│  - sentence-transformers 或   │
│    OpenAI text-embedding      │
│  - 每段對話 → 一個向量         │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Step 3: BERTopic 分群        │
│  - UMAP 降維                  │
│  - HDBSCAN 密度聚類           │
│  - c-TF-IDF 主題關鍵字        │
│  - 自動決定群數 + outlier 標記 │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Step 4: LLM FAQ 提取         │
│  - 每群取 top-K 代表性對話    │
│  - LLM 提取 Q&A pairs        │
│  - 偵測同群內的矛盾資訊       │
│  - 標註出現頻率               │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Step 5: 輸出 + 人工抽查      │
│  - 輸出 markdown 到 data/     │
│  - 矛盾標記讓人工確認         │
│  - 視覺化 topic map           │
└──────────────────────────────┘
    │
    ▼
  KBee ingest → RAG 可查詢
```

---

## 技術選型

| 元件 | 選擇 | 理由 |
|------|------|------|
| Clustering | **BERTopic** (17k+ ⭐) | 成熟、embedding-based、自動決群數、有視覺化 |
| Embedding | **OpenAI text-embedding-3-small** | KBee 已在用，不加新依賴；或用 sentence-transformers 省 API 費 |
| 降維 | **UMAP**（BERTopic 內建） | 保留語義距離，比 PCA 好 |
| 聚類 | **HDBSCAN**（BERTopic 內建） | 密度聚類、自動 outlier、不需預設 K |
| FAQ 提取 | **OpenAI GPT** | KBee 已在用 |
| 矛盾偵測 | **LLM prompt** | 同群內比對多段對話，標記不一致處 |

### 不採用的方案

| 方案 | 不採用理由 |
|------|-----------|
| LangChain | KBee 不依賴它，加框架 = 加維護成本 |
| LLM 直接分群 | 不穩定、不可重現、大量對話塞不進 context |
| Microsoft Conversation Knowledge Mining | 綁 Azure 全家桶，不適合獨立產品 |
| Forethought / Assembled / Loris | 閉源 SaaS，enterprise 定價 |

---

## 輸入格式（待定）

預期支援：

```json
// 格式 A：JSON array
[
  {
    "conversation_id": "001",
    "messages": [
      {"role": "customer", "content": "我想退貨"},
      {"role": "agent", "content": "請提供訂單編號"}
    ],
    "metadata": {
      "date": "2025-12-01",
      "agent_name": "Alice",
      "channel": "LINE"
    }
  }
]
```

```csv
# 格式 B：CSV
conversation_id,role,content,timestamp
001,customer,我想退貨,2025-12-01T10:00:00
001,agent,請提供訂單編號,2025-12-01T10:01:00
```

---

## 輸出格式

每個 topic 生成一個 markdown 檔案，放在 `data/auto_faq/` 下：

```markdown
# 退貨與退款

> 自動從 152 段對話中提取（2025-12-01 ~ 2026-01-31）
> Topic cluster #3 | 代表關鍵字：退貨、退款、寄回、瑕疵、期限

## 常見問題

### Q1: 商品壞了/不能用，要怎麼退貨？
**A:** 請提供訂單編號，我們會幫您查詢。若商品在 7 天退貨期限內，
會直接為您開立退貨單。商品寄回後 3-5 個工作天完成退款。

### Q2: 超過退貨期限還能退嗎？
**A:** 一般退貨期限為 7 天。VIP 會員享有 30 天退貨期限。

## 關鍵規則
- 退貨期限：收到商品後 7 天（VIP 30 天）
- 退款時間：收到退貨後 3-5 個工作天
- 退款方式：原付款方式退回

## ⚠️ 矛盾/待確認
- 對話 #87 中客服說「10 天內可退」，與多數對話的「7 天」不一致
  → 需人工確認哪個是正確政策
```

---

## TODO

### Phase 1: 基礎 pipeline（MVP）
- [ ] 建立 `scripts/conversation_mining.py` 主腳本
- [ ] 實作 Step 1：前處理（支援 JSON + CSV 輸入）
- [ ] 實作 Step 2：Embedding（用 OpenAI text-embedding-3-small）
- [ ] 實作 Step 3：BERTopic clustering
- [ ] 實作 Step 4：LLM FAQ 提取 prompt
- [ ] 實作 Step 5：Markdown 輸出到 `data/auto_faq/`
- [ ] 加入 `pyproject.toml` 依賴：`bertopic`, `hdbscan`, `umap-learn`
- [ ] 基本 CLI 介面：`uv run python scripts/conversation_mining.py --input data.json --output data/auto_faq/`

### Phase 2: 品質強化
- [ ] 矛盾偵測：同 cluster 內比對多段對話，標記不一致
- [ ] 視覺化：輸出 BERTopic topic map（HTML interactive）
- [ ] 支援增量更新：新對話進來時只處理差異，不重跑全部
- [ ] 多語言支援：同時處理中英文對話

### Phase 3: 產品化
- [ ] 整合進 KBee ingest pipeline（`kbee ingest --from-conversations`）
- [ ] Web UI：上傳對話檔 → 預覽提取結果 → 確認後 ingest
- [ ] 自動排程：定期從客服系統拉新對話 → 更新 FAQ
- [ ] 品質評分：用 eval 框架衡量自動生成的 FAQ vs 人工 FAQ

### 延伸（層次 3 — 學互動策略）
- [ ] 從對話中提取「應對模式」（先安撫 → 再解釋 → 再給方案）
- [ ] 用 DSPy 或 fine-tuning 讓回覆風格更像真人客服
- [ ] 客服品質評分：標記好/壞對話，作為 RLHF 訓練資料

---

## 參考資源

- [BERTopic 官方文件](https://maartengr.github.io/BERTopic/)
- [BERTopic GitHub](https://github.com/MaartenGr/BERTopic)（17k+ ⭐）
- [Microsoft Conversation Knowledge Mining](https://github.com/microsoft/conversation-knowledge-mining-solution-accelerator)（參考架構）
- [HDBSCAN 文件](https://hdbscan.readthedocs.io/)
- [UMAP 文件](https://umap-learn.readthedocs.io/)

---

## 決策紀錄

| 日期 | 決策 | 理由 |
|------|------|------|
| 2026-02-14 | 用 BERTopic (embedding + HDBSCAN) 做 clustering，不用 LLM 直接分群 | LLM 分群不穩定、不可重現；embedding clustering 可量化、可視覺化 |
| 2026-02-14 | 不用 LangChain | KBee 不依賴它，直接用 OpenAI SDK 更輕量 |
| 2026-02-14 | 不用 Azure Conversation Knowledge Mining | 綁 Azure 全家桶，KBee 要保持平台獨立 |
| 2026-02-14 | chunk_size 從 1024 改為 512，overlap 從 200 改為 50 | 20 組參數實驗結果：512/50 是最小 chunk_size 達到 100% retrieval accuracy |
