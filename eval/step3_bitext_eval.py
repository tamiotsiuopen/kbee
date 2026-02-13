"""Step 3: Comprehensive evaluation on Bitext FAQ (Traditional Chinese).

For each of the 27 intents, picks 2 question variants from the original
dataset (translated to Chinese via the LLM), queries KBee, and checks:
1. Retrieval: did we find the right intent's FAQ?
2. Generation: is the answer relevant and not hallucinated?
"""

import json
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import OpenAI
from datasets import load_dataset

client = OpenAI()

# Map intent -> expected FAQ file index (1-based, matching build order)
INTENT_ORDER = [
    "cancel_order", "change_order", "change_shipping_address",
    "check_cancellation_fee", "check_invoice", "check_payment_methods",
    "check_refund_policy", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account",
    "get_invoice", "get_refund", "newsletter_subscription",
    "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address",
    "switch_account", "track_order", "track_refund",
]

INTENT_ZH_KEYWORDS = {
    "cancel_order": ["取消", "訂單"],
    "change_order": ["更改", "修改", "訂單"],
    "change_shipping_address": ["更改", "運送", "地址"],
    "check_cancellation_fee": ["取消", "費用"],
    "check_invoice": ["查看", "發票"],
    "check_payment_methods": ["付款", "方式"],
    "check_refund_policy": ["退款", "政策"],
    "complaint": ["投訴", "申訴", "索賠"],
    "contact_customer_service": ["聯繫", "客服"],
    "contact_human_agent": ["真人", "客服", "人工"],
    "create_account": ["建立", "註冊", "帳戶"],
    "delete_account": ["刪除", "帳戶"],
    "delivery_options": ["配送", "方式", "選項"],
    "delivery_period": ["配送", "時間", "送達"],
    "edit_account": ["編輯", "修改", "帳戶"],
    "get_invoice": ["取得", "發票"],
    "get_refund": ["退款", "申請"],
    "newsletter_subscription": ["訂閱", "電子報"],
    "payment_issue": ["付款", "問題"],
    "place_order": ["下單", "訂購"],
    "recover_password": ["密碼", "重設", "找回"],
    "registration_problems": ["註冊", "問題"],
    "review": ["評價", "評論"],
    "set_up_shipping_address": ["設定", "運送", "地址"],
    "switch_account": ["切換", "帳戶"],
    "track_order": ["追蹤", "訂單"],
    "track_refund": ["追蹤", "退款"],
}


def translate_question(en_question: str, intent: str) -> str:
    """Translate a single English query to natural Traditional Chinese.

    We pass the expected intent to reduce translation noise (slang/typos)
    that could otherwise distort the meaning and invalidate retrieval eval.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": (
                "You translate customer-support user queries to Traditional Chinese (Taiwan). "
                "Preserve the user's intent even if the English is slangy/typo/ambiguous. "
                "Do NOT invent new topics. Keep it short."
            ),
        }, {
            "role": "user",
            "content": (
                f"Expected intent: {intent}\n"
                f"English query: {en_question}\n\n"
                "Return only the Traditional Chinese query."
            ),
        }],
        temperature=0,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


def judge_answer(question: str, answer: str, intent: str) -> dict:
    """Use LLM to judge answer quality."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "You are an evaluation judge. Score the answer on a 1-5 scale for each metric. Output JSON only."
        }, {
            "role": "user",
            "content": f"""Evaluate this customer service chatbot response.

Question: {question}
Expected topic: {intent}
Answer: {answer}

Score each metric (1=terrible, 5=excellent):
- relevance: Does the answer address the question topic?
- helpfulness: Does the answer provide actionable information?
- accuracy: Is the answer factually consistent (no hallucination)?
- conciseness: Is the answer appropriately brief?

Output format: {{"relevance": N, "helpfulness": N, "accuracy": N, "conciseness": N}}"""
        }],
        temperature=0,
        max_tokens=100,
    )
    try:
        text = resp.choices[0].message.content.strip()
        # Extract JSON from possible markdown
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception:
        return {"relevance": 0, "helpfulness": 0, "accuracy": 0, "conciseness": 0}


def run_evaluation():
    from kbee.query import get_query_engine

    print("=" * 60)
    print("STEP 3: Bitext FAQ Comprehensive Evaluation")
    print("=" * 60)

    # Resume support
    resume_path = Path("eval/results_step3_bitext.partial.json")
    done_keys = set()
    results = []
    all_scores = {"relevance": [], "helpfulness": [], "accuracy": [], "conciseness": []}
    retrieval_correct = 0
    total = 0

    if resume_path.exists():
        data = json.loads(resume_path.read_text(encoding="utf-8"))
        results = data.get("details", [])
        for r in results:
            done_keys.add(f"{r['intent']}|{r['question_en']}")
            total += 1
            if r.get("retrieval_ok"):
                retrieval_correct += 1
            sc = r.get("scores", {})
            for m in all_scores:
                all_scores[m].append(sc.get(m, 0))
        print(f"Resuming from partial results: {total} already done")

    # Load original dataset for question variants
    print("\nLoading Bitext dataset for question variants...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    intent_questions = {}
    for row in ds:
        intent = row["intent"]
        if intent not in intent_questions:
            intent_questions[intent] = []
        intent_questions[intent].append(row["instruction"])

    engine = get_query_engine()

    print(f"\nTesting 2 questions per intent × {len(INTENT_ORDER)} intents = {len(INTENT_ORDER)*2} questions\n")

    for intent in INTENT_ORDER:
        # Pick 2 diverse questions (index 0 and ~middle)
        qs_en = intent_questions[intent]
        picks = [qs_en[0], qs_en[len(qs_en) // 2]]

        for q_en in picks:
            key = f"{intent}|{q_en}"
            if key in done_keys:
                continue
            total += 1
            # Translate
            q_zh = translate_question(q_en, intent)

            # Query
            response = engine.query(q_zh)
            answer = str(response)

            # Check retrieval via keywords
            contexts = []
            if hasattr(response, "source_nodes"):
                contexts = [node.text for node in response.source_nodes]
            all_context = " ".join(contexts)

            keywords = INTENT_ZH_KEYWORDS.get(intent, [])
            kw_found = sum(1 for kw in keywords if kw in all_context) if keywords else 0
            kw_total = len(keywords)
            retrieval_ok = (kw_found / kw_total >= 0.5) if kw_total > 0 else True
            if retrieval_ok:
                retrieval_correct += 1

            # Judge answer quality
            scores = judge_answer(q_zh, answer, intent)

            result = {
                "intent": intent,
                "question_en": q_en,
                "question_zh": q_zh,
                "answer": answer,
                "retrieval_ok": retrieval_ok,
                "retrieval_keywords": f"{kw_found}/{kw_total}",
                "scores": scores,
            }
            results.append(result)

            for metric in all_scores:
                all_scores[metric].append(scores.get(metric, 0))

            status = "✅" if retrieval_ok else "❌"
            print(f"  [{total:2d}/54] {intent:30s} | Ret: {status} | "
                  f"Rel:{scores.get('relevance',0)} Help:{scores.get('helpfulness',0)} "
                  f"Acc:{scores.get('accuracy',0)} Con:{scores.get('conciseness',0)} | "
                  f"Q: {q_zh[:30]}...")

            # Save partial after each question
            partial = {
                "total_questions": len(INTENT_ORDER) * 2,
                "done": len(results),
                "retrieval_correct": retrieval_correct,
                "avg_scores": {
                    m: (round(sum(v)/len(v), 2) if v else 0) for m, v in all_scores.items()
                },
                "details": results,
            }
            resume_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Bitext FAQ Evaluation (54 questions)")
    print("=" * 60)

    print(f"\nRetrieval accuracy: {retrieval_correct}/{total} ({retrieval_correct/total:.1%})")
    print(f"\nGeneration quality (1-5 scale):")
    for metric, scores_list in all_scores.items():
        avg = sum(scores_list) / len(scores_list) if scores_list else 0
        print(f"  {metric:15s}: {avg:.2f}")

    overall = sum(sum(v) for v in all_scores.values()) / (len(all_scores) * total) if total else 0
    print(f"\n  {'Overall':15s}: {overall:.2f}")

    # Distribution
    print(f"\nScore distribution (relevance):")
    for score in range(1, 6):
        count = all_scores["relevance"].count(score)
        bar = "█" * count
        print(f"  {score}: {bar} ({count})")

    # Save
    output_path = Path("eval/results_step3_bitext.json")
    summary = {
        "total_questions": total,
        "retrieval_accuracy": f"{retrieval_correct}/{total} ({retrieval_correct/total:.1%})",
        "avg_scores": {m: round(sum(v)/len(v), 2) for m, v in all_scores.items()},
        "overall_avg": round(overall, 2),
        "details": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
