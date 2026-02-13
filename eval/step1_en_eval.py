"""Step 1: English pipeline validation using ragas.

Uses a small English FAQ dataset to verify the RAG pipeline's
retrieval and generation quality before testing Chinese.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kbee.config import settings
from kbee.ingest import ingest_documents, get_chroma_client


def create_english_test_data():
    """Create English FAQ test documents."""
    data_dir = Path("eval/data_en")
    data_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        ("refund_policy.txt", """Refund Policy

Q: How do I request a refund?
A: You can request a refund within 30 days of purchase. Go to "My Orders", select the order, and click "Request Refund". Refunds are processed within 5-7 business days to your original payment method.

Q: Can I get a refund after 30 days?
A: After the 30-day window, refunds are only available for defective products. Please contact customer support with photos of the defect.

Q: How long does it take to receive my refund?
A: Standard refunds take 5-7 business days. Credit card refunds may take an additional 1-2 billing cycles to appear on your statement."""),

        ("shipping.txt", """Shipping Information

Q: How long does shipping take?
A: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. Remote areas may require an additional 2-3 business days.

Q: Do you ship internationally?
A: Yes, we ship to Japan, Korea, Hong Kong, Singapore, and the United States. International shipping takes 7-14 business days. Some items may be restricted due to regulations.

Q: How much does shipping cost?
A: Standard shipping is free for orders over $50. Express shipping costs $9.99. International shipping rates vary by destination and weight."""),

        ("account.txt", """Account Management

Q: How do I create an account?
A: Click "Sign Up" in the top right corner. Enter your email, phone number, and password. You can also sign up with Google, Facebook, or Apple ID.

Q: How do I reset my password?
A: Go to Settings > Security > Change Password. Enter your current password and new password. We recommend using at least 8 characters with uppercase, lowercase, and numbers.

Q: How do I delete my account?
A: Contact customer support to request account deletion. Please note that this action is irreversible and all order history will be lost."""),

        ("membership.txt", """VIP Membership Program

Q: How do I become a VIP member?
A: When your annual spending reaches $300, you will automatically be upgraded to VIP status. VIP members enjoy 10% discount, free shipping, and priority customer support.

Q: What are the benefits of VIP membership?
A: VIP members get: 10% discount on all orders, free standard shipping, priority customer support, double loyalty points, and early access to sales events.

Q: Do VIP benefits expire?
A: VIP status is valid for one year from the date of qualification. You need to maintain $300 annual spending to keep your VIP status."""),

        ("returns.txt", """Return Policy

Q: How do I return a product?
A: Initiate a return within 14 days of delivery through "My Orders". Pack the item in its original packaging and attach the return label. Drop it off at any designated courier point.

Q: What items cannot be returned?
A: Personalized items, perishable goods, digital products, and items marked as "final sale" cannot be returned. Opened software and hygiene products are also non-returnable.

Q: Who pays for return shipping?
A: If the return is due to a defect or our error, we cover shipping costs. For other returns, the customer is responsible for return shipping fees."""),
    ]

    for filename, content in docs:
        (data_dir / filename).write_text(content)

    print(f"Created {len(docs)} English test documents in {data_dir}")
    return str(data_dir)


# Ground truth test cases: (question, expected_answer, expected_source_keywords)
TEST_CASES = [
    {
        "question": "How do I request a refund?",
        "ground_truth": "You can request a refund within 30 days of purchase by going to 'My Orders', selecting the order, and clicking 'Request Refund'. Refunds are processed within 5-7 business days.",
        "source_keywords": ["refund", "30 days"],
    },
    {
        "question": "What is the shipping time for express delivery?",
        "ground_truth": "Express shipping takes 1-2 business days.",
        "source_keywords": ["express", "1-2 business days"],
    },
    {
        "question": "How can I sign up for an account?",
        "ground_truth": "Click 'Sign Up' in the top right corner, enter your email, phone number, and password. You can also sign up with Google, Facebook, or Apple ID.",
        "source_keywords": ["sign up", "email"],
    },
    {
        "question": "What are VIP member benefits?",
        "ground_truth": "VIP members get 10% discount on all orders, free standard shipping, priority customer support, double loyalty points, and early access to sales events.",
        "source_keywords": ["10%", "free shipping", "priority"],
    },
    {
        "question": "Can I return opened software?",
        "ground_truth": "No, opened software and hygiene products are non-returnable.",
        "source_keywords": ["non-returnable", "software"],
    },
    {
        "question": "Do you ship to Europe?",
        "ground_truth": "The knowledge base does not mention shipping to Europe. Currently supported international destinations are Japan, Korea, Hong Kong, Singapore, and the United States.",
        "source_keywords": ["Japan", "Korea", "United States"],
    },
    {
        "question": "How much annual spending is needed for VIP?",
        "ground_truth": "Annual spending of $300 is needed to qualify for VIP membership, and this amount must be maintained annually to keep VIP status.",
        "source_keywords": ["$300", "annual"],
    },
    {
        "question": "Who pays return shipping for defective items?",
        "ground_truth": "For defective items or errors on the company's part, the company covers return shipping costs.",
        "source_keywords": ["defect", "cover", "shipping"],
    },
    {
        "question": "Can I get a refund after 60 days?",
        "ground_truth": "After the 30-day window, refunds are only available for defective products.",
        "source_keywords": ["30-day", "defective"],
    },
    {
        "question": "What is your company's stock price?",
        "ground_truth": "This information is not available in the knowledge base.",
        "source_keywords": [],
    },
]


def run_evaluation():
    """Run Step 1 evaluation."""
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    # 1. Create test data and ingest
    print("=" * 60)
    print("STEP 1: English Pipeline Validation")
    print("=" * 60)

    data_dir = create_english_test_data()

    # Use a separate collection for English eval
    settings_chroma = settings.chroma_path
    settings_chroma.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings_chroma))

    # Clear and recreate eval collection
    try:
        client.delete_collection("kbee_eval_en")
    except Exception:
        pass
    collection = client.get_or_create_collection("kbee_eval_en")

    # Ingest using llama_index directly (to use separate collection)
    from llama_index.core import SimpleDirectoryReader, StorageContext
    from llama_index.core.node_parser import SentenceSplitter

    reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
    documents = reader.load_data()
    print(f"\nLoaded {len(documents)} documents")

    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )
    print(f"Ingested into eval collection. Count: {collection.count()}")

    # 2. Build query engine (English mode - no Chinese system prompt)
    llm = OpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        system_prompt=(
            "You are a helpful customer service assistant. "
            "Answer questions ONLY based on the provided context. "
            "If the answer is not in the context, say: "
            "'This information is not available in the knowledge base.'"
        ),
    )

    query_engine = index.as_query_engine(
        llm=llm,
        embed_model=embed_model,
        similarity_top_k=3,
    )

    # 3. Run queries and collect results
    print("\n" + "=" * 60)
    print("Running queries...")
    print("=" * 60)

    results = []
    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        response = query_engine.query(q)

        # Extract retrieved contexts
        contexts = []
        if hasattr(response, "source_nodes"):
            contexts = [node.text for node in response.source_nodes]

        result = {
            "question": q,
            "answer": str(response),
            "ground_truth": tc["ground_truth"],
            "contexts": contexts,
            "source_keywords": tc["source_keywords"],
        }
        results.append(result)

        # Simple keyword-based retrieval check
        all_context = " ".join(contexts).lower()
        keywords_found = sum(
            1 for kw in tc["source_keywords"]
            if kw.lower() in all_context
        )
        keyword_total = len(tc["source_keywords"])
        retrieval_ok = keywords_found == keyword_total if keyword_total > 0 else True

        print(f"\n--- Q{i+1}: {q}")
        print(f"  Answer: {str(response)[:150]}...")
        print(f"  Retrieval: {'✅' if retrieval_ok else '❌'} ({keywords_found}/{keyword_total} keywords found)")

    # 4. Summary metrics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    retrieval_scores = []
    for r in results:
        all_context = " ".join(r["contexts"]).lower()
        if len(r["source_keywords"]) > 0:
            score = sum(1 for kw in r["source_keywords"] if kw.lower() in all_context) / len(r["source_keywords"])
        else:
            score = 1.0
        retrieval_scores.append(score)

    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
    perfect_retrieval = sum(1 for s in retrieval_scores if s == 1.0)

    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Avg retrieval keyword match: {avg_retrieval:.1%}")
    print(f"Perfect retrieval: {perfect_retrieval}/{len(TEST_CASES)}")

    # Save detailed results
    output_path = Path("eval/results_step1_en.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")

    # Cleanup
    try:
        client.delete_collection("kbee_eval_en")
    except Exception:
        pass

    return results


if __name__ == "__main__":
    run_evaluation()
