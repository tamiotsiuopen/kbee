"""Build Traditional Chinese FAQ knowledge base from Bitext dataset.

Takes the Bitext customer support dataset (English), extracts canonical
Q&A per intent, translates to Traditional Chinese via OpenAI, and
generates FAQ files for KBee ingestion.
"""

import json
import os
import sys
from pathlib import Path
from openai import OpenAI

from datasets import load_dataset

OUTPUT_DIR = Path("data/bitext_faq_zh")
SAMPLE_QUESTIONS_PER_INTENT = 5  # Include a few question variants

client = OpenAI()


def translate_intent(category: str, intent: str, questions: list[str], response: str) -> str:
    """Translate a single intent's FAQ to Traditional Chinese."""
    # Pick diverse questions
    sample_qs = questions[:SAMPLE_QUESTIONS_PER_INTENT]
    
    prompt = f"""Translate the following customer service FAQ entry into Traditional Chinese (台灣繁體中文).
Make it sound natural for a Taiwanese e-commerce context. 
Replace placeholder tokens like {{{{Order Number}}}}, {{{{Account ID}}}} etc. with realistic examples in parentheses.

Category: {category}
Intent: {intent}

Sample customer questions:
{chr(10).join(f'- {q}' for q in sample_qs)}

Agent response:
{response}

Output format (in Traditional Chinese):
分類：[category in Chinese]

常見問法：
- [translated question variant 1]
- [translated question variant 2]
- [translated question variant 3]
- [translated question variant 4]
- [translated question variant 5]

標準回答：
[translated response, natural Taiwanese Chinese, ~2-4 sentences]
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def main():
    print("Loading Bitext dataset...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )

    # Group by intent
    intent_data = {}
    for row in ds:
        intent = row["intent"]
        if intent not in intent_data:
            intent_data[intent] = {
                "category": row["category"],
                "intent": intent,
                "questions": [],
                "response": row["response"],
            }
        intent_data[intent]["questions"].append(row["instruction"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTranslating {len(intent_data)} intents to Traditional Chinese...")
    all_faqs = []

    for i, (intent, data) in enumerate(intent_data.items()):
        print(f"  [{i+1}/{len(intent_data)}] {data['category']}/{intent}...", end=" ", flush=True)
        
        translated = translate_intent(
            data["category"],
            intent,
            data["questions"],
            data["response"],
        )
        
        # Save individual file
        filename = f"{i+1:02d}_{intent}.txt"
        (OUTPUT_DIR / filename).write_text(translated, encoding="utf-8")
        all_faqs.append(translated)
        print("✅")

    # Save combined file
    combined = "\n\n---\n\n".join(all_faqs)
    (OUTPUT_DIR / "all_faq_zh.txt").write_text(combined, encoding="utf-8")

    print(f"\nDone! {len(all_faqs)} FAQ files saved to {OUTPUT_DIR}/")
    print(f"Combined file: {OUTPUT_DIR}/all_faq_zh.txt")


if __name__ == "__main__":
    main()
