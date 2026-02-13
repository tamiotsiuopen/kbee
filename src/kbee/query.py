"""RAG query engine for KBee.

Loads the ChromaDB index and provides a query interface
that retrieves relevant documents and generates responses.
"""

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from kbee.config import settings

SYSTEM_PROMPT = (
    "You are KBee, a helpful customer service AI assistant. "
    "Language policy: always reply in Traditional Chinese. "
    "Do not repeatedly introduce yourself unless the user explicitly asks who "
    "you are. "
    "Answer questions ONLY based on the provided knowledge base content. "
    "Use this response structure only when it helps clarity for non-trivial "
    "questions: 1) 先給結論，2) 補充必要的關鍵細節，3) 提供下一步建議。 "
    "For greetings or simple chit-chat, reply naturally in 1-2 short sentences. "
    "Do not output rigid labels like '結論' or '下一步建議' unless the user asks "
    "for a structured format. "
    "If user information is insufficient or ambiguous, ask one key clarifying "
    "question first instead of guessing. "
    "Do not make promises about refunds, compensation, legal outcomes, or any "
    "policy commitments unless they are explicitly stated in the knowledge base. "
    "If the answer is not found in the knowledge base, say: "
    "'我的知識庫裡沒有這方面的資訊，請聯絡客服團隊取得進一步協助。' "
    "Always be polite, empathetic, and helpful. "
    "Keep replies as brief as possible unless more detail is necessary. "
    "When possible, cite which document your answer comes from."
)


def get_query_engine() -> BaseQueryEngine:
    """Create and return a query engine backed by ChromaDB.

    Returns:
        A LlamaIndex query engine configured with the ChromaDB
        vector store, OpenAI embedding, and LLM.

    Raises:
        FileNotFoundError: If the ChromaDB storage directory
            does not exist.
    """
    if not settings.chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB storage not found at: {settings.chroma_path}. "
            "Run ingestion first: python -m kbee.ingest"
        )

    # Connect to existing ChromaDB collection.
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(name=settings.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Set up models.
    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    llm = OpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        system_prompt=SYSTEM_PROMPT,
    )

    # Build index from existing vector store.
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=settings.similarity_top_k,
    )


async def query_knowledge_base(
    query_engine: BaseQueryEngine,
    question: str,
) -> RESPONSE_TYPE:
    """Query the knowledge base and return a response.

    Args:
        query_engine: The LlamaIndex query engine to use.
        question: The user's question.

    Returns:
        The query response including answer text and source nodes.
    """
    return await query_engine.aquery(question)
