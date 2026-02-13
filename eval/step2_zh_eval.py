"""Step 2: Chinese RAG evaluation with ground truth.

Tests the actual production configuration (Chinese FAQ + Chinese system prompt).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kbee.config import settings

# Chinese test cases against the existing FAQ data in data/faq/zh_tw/
TEST_CASES = [
    # --- Direct match questions ---
    {
        "question": "請問如何申請退款？",
        "ground_truth": "購買後 30 天內可申請退款。前往「我的訂單」選擇訂單，點擊「申請退款」，退款在 5-7 個工作天內退回原付款方式。",
        "source_keywords": ["30 天", "我的訂單", "申請退款"],
    },
    {
        "question": "你們有哪些付款方式？",
        "ground_truth": "支援信用卡（Visa、MasterCard、JCB）、銀行轉帳、超商付款、LINE Pay、Apple Pay 和 Google Pay。",
        "source_keywords": ["Visa", "LINE Pay", "Apple Pay"],
    },
    {
        "question": "VIP 會員有什麼優惠？",
        "ground_truth": "VIP 會員享有 9 折優惠、免運費、專屬客服通道。年消費達 NT$10,000 自動升級。",
        "source_keywords": ["9 折", "免運費", "10,000"],
    },
    # --- Paraphrased / colloquial questions ---
    {
        "question": "東西壞了可以換嗎？",
        "ground_truth": "收到瑕疵品 7 天內聯繫客服並提供照片，確認後免費退換貨，運費由公司承擔。",
        "source_keywords": ["瑕疵", "7 天", "照片"],
    },
    {
        "question": "怎麼樣才能變 VIP？",
        "ground_truth": "年度消費金額達到 NT$10,000 時系統自動升級為 VIP 會員。",
        "source_keywords": ["10,000", "自動升級"],
    },
    {
        "question": "我想取消訂單怎麼辦？",
        "ground_truth": "未出貨可在「我的訂單」直接取消。已出貨請聯繫客服安排退貨。取消後退款 3-5 個工作天處理。",
        "source_keywords": ["取消", "我的訂單", "3-5"],
    },
    # --- Multi-intent / complex questions ---
    {
        "question": "我想退款順便問一下運費多少？",
        "ground_truth": "退款：購買後 30 天內申請。配送：標準配送 3-5 天，快速配送 1-2 天。",
        "source_keywords": ["30 天", "配送"],
    },
    {
        "question": "積分怎麼用？VIP 有額外積分嗎？",
        "ground_truth": "每消費 NT$1 累積 1 點，100 點折抵 NT$1，有效期 1 年。VIP 享雙倍積分。",
        "source_keywords": ["積分", "100 點", "雙倍"],
    },
    # --- Edge cases ---
    {
        "question": "可以寄到法國嗎？",
        "ground_truth": "目前支援配送至日本、韓國、香港、新加坡和美國，未提及法國。",
        "source_keywords": ["日本", "美國"],
    },
    {
        "question": "你們的 CEO 是誰？",
        "ground_truth": "知識庫中沒有這方面的資訊。",
        "source_keywords": [],
    },
    # --- Typo / informal ---
    {
        "question": "密碼忘了怎辦",
        "ground_truth": "前往設定頁面 > 安全設定 > 更改密碼。需輸入目前密碼和新密碼，建議至少 8 個字元含大小寫和數字。",
        "source_keywords": ["安全設定", "更改密碼"],
    },
    {
        "question": "APP 哪裡下載？",
        "ground_truth": "在 App Store 或 Google Play 搜尋品牌名稱下載。首次下載可獲 NT$100 折扣券。",
        "source_keywords": ["App Store", "Google Play", "100"],
    },
]


def run_evaluation():
    """Run Step 2 Chinese evaluation."""
    from kbee.query import get_query_engine

    print("=" * 60)
    print("STEP 2: Chinese RAG Evaluation")
    print("=" * 60)

    # Use the existing Chinese FAQ data (already ingested)
    engine = get_query_engine()

    results = []
    retrieval_scores = []
    
    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        response = engine.query(q)

        contexts = []
        if hasattr(response, "source_nodes"):
            contexts = [node.text for node in response.source_nodes]

        # Keyword retrieval check
        all_context = " ".join(contexts)
        kw_total = len(tc["source_keywords"])
        if kw_total > 0:
            kw_found = sum(1 for kw in tc["source_keywords"] if kw in all_context)
            score = kw_found / kw_total
        else:
            kw_found = 0
            score = 1.0
        retrieval_scores.append(score)

        result = {
            "question": q,
            "answer": str(response),
            "ground_truth": tc["ground_truth"],
            "contexts": contexts,
            "retrieval_score": score,
        }
        results.append(result)

        status = "✅" if score == 1.0 else ("🟡" if score >= 0.5 else "❌")
        print(f"\n--- Q{i+1}: {q}")
        print(f"  Answer: {str(response)[:200]}...")
        print(f"  Retrieval: {status} ({kw_found}/{kw_total} keywords)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Chinese RAG Evaluation")
    print("=" * 60)

    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
    perfect = sum(1 for s in retrieval_scores if s == 1.0)
    partial = sum(1 for s in retrieval_scores if 0 < s < 1.0)
    failed = sum(1 for s in retrieval_scores if s == 0)

    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Avg retrieval score: {avg_retrieval:.1%}")
    print(f"Perfect retrieval: {perfect}/{len(TEST_CASES)}")
    print(f"Partial retrieval: {partial}/{len(TEST_CASES)}")
    print(f"Failed retrieval: {failed}/{len(TEST_CASES)}")

    # Save
    output_path = Path("eval/results_step2_zh.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
