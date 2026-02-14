"""Fast parameter search - retrieval-only, no LLM generation, cached translations."""
import json
import sys
import os
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CHUNK_SIZES = [256, 512, 768, 1024, 1536]
CHUNK_OVERLAPS = [0, 50, 100, 200]
PROJECT_DIR = Path(__file__).parent.parent
CACHE_FILE = PROJECT_DIR / "eval" / "translations_cache.json"

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


def get_translations():
    """Get cached translations or translate once."""
    if CACHE_FILE.exists():
        print("  Using cached translations")
        return json.loads(CACHE_FILE.read_text())
    
    from openai import OpenAI
    from datasets import load_dataset
    
    client = OpenAI()
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    
    intent_questions = {}
    for row in ds:
        intent = row["intent"]
        if intent not in intent_questions:
            intent_questions[intent] = []
        intent_questions[intent].append(row["instruction"])
    
    translations = []
    for intent in INTENT_ORDER:
        qs = intent_questions[intent]
        picks = [qs[0], qs[len(qs) // 2]]
        for q_en in picks:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You translate customer-support user queries to Traditional Chinese (Taiwan). Keep it short."
                }, {
                    "role": "user",
                    "content": f"Expected intent: {intent}\nEnglish query: {q_en}\n\nReturn only the Traditional Chinese query."
                }],
                temperature=0, max_tokens=80,
            )
            q_zh = resp.choices[0].message.content.strip()
            translations.append({"intent": intent, "q_en": q_en, "q_zh": q_zh})
            print(f"  Translated: {q_zh[:40]}...")
    
    CACHE_FILE.write_text(json.dumps(translations, indent=2, ensure_ascii=False))
    return translations


def eval_combo(chunk_size, chunk_overlap, translations):
    """Ingest + retrieval-only eval for one combo."""
    import chromadb
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from kbee.config import settings
    
    storage_dir = str(PROJECT_DIR / f"storage_{chunk_size}_{chunk_overlap}")
    
    # Ingest
    print(f"  Ingesting...")
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=storage_dir)
    try:
        client.delete_collection("kbee_docs")
    except Exception:
        pass
    collection = client.get_or_create_collection("kbee_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    
    reader = SimpleDirectoryReader(
        input_dir=settings.data_dir,
        recursive=True,
        required_exts=[".pdf", ".txt", ".docx", ".md", ".html"],
    )
    documents = reader.load_data()
    
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=False,
    )
    
    num_chunks = len(index.docstore.docs)
    print(f"  Chunks: {num_chunks}")
    
    # Retrieval only (no LLM)
    retriever = index.as_retriever(similarity_top_k=settings.similarity_top_k)
    
    correct = 0
    total = len(translations)
    failures = []
    
    for i, t in enumerate(translations):
        nodes = retriever.retrieve(t["q_zh"])
        all_context = " ".join(n.text for n in nodes)
        
        keywords = INTENT_ZH_KEYWORDS.get(t["intent"], [])
        kw_found = sum(1 for kw in keywords if kw in all_context)
        kw_total = len(keywords)
        ok = (kw_found / kw_total >= 0.5) if kw_total > 0 else True
        if ok:
            correct += 1
        else:
            failures.append(t["intent"])
        
        if not ok:
            print(f"    ❌ [{i+1}] {t['intent']} ({kw_found}/{kw_total} keywords)")
    
    accuracy = correct / total
    print(f"  Result: {correct}/{total} ({accuracy:.1%})")
    
    # Cleanup chroma client
    del client
    
    return {"correct": correct, "total": total, "accuracy_pct": round(accuracy * 100, 1), "chunks": num_chunks, "failures": failures}


def main():
    results_file = PROJECT_DIR / "eval" / "param_search_results.json"
    
    if results_file.exists():
        all_results = json.loads(results_file.read_text())
    else:
        all_results = {}
    
    print("Step 1: Getting translations...")
    translations = get_translations()
    print(f"  {len(translations)} queries ready\n")
    
    for cs, co in product(CHUNK_SIZES, CHUNK_OVERLAPS):
        if co >= cs:
            continue
        key = f"{cs}_{co}"
        if key in all_results:
            print(f"Skipping {cs}/{co} (already: {all_results[key]['accuracy_pct']}%)")
            continue
        
        print(f"\n{'='*50}")
        print(f"chunk_size={cs}, chunk_overlap={co}")
        print(f"{'='*50}")
        
        result = eval_combo(cs, co, translations)
        all_results[key] = result
        results_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    
    # Summary
    print("\n\n" + "="*70)
    print("PARAMETER SEARCH RESULTS")
    print("="*70)
    print(f"{'chunk_size':>12} {'overlap':>8} {'accuracy':>10} {'chunks':>8}")
    print("-"*45)
    for cs in CHUNK_SIZES:
        for co in CHUNK_OVERLAPS:
            key = f"{cs}_{co}"
            if key in all_results:
                r = all_results[key]
                print(f"{cs:>12} {co:>8} {r['accuracy_pct']:>9.1f}% {r['chunks']:>8}")
    
    best_key = max(all_results, key=lambda k: (all_results[k]['correct'], -all_results[k].get('chunks', 999)))
    best = all_results[best_key]
    cs, co = best_key.split("_")
    print(f"\nBest: chunk_size={cs}, chunk_overlap={co} → {best['accuracy_pct']}%, {best['chunks']} chunks")


if __name__ == "__main__":
    main()
