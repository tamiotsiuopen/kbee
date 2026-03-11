"""RAG query module for KBee.

Provides two modes:
- Retriever-only (for realtime voice): fast vector retrieval, no LLM
- Full query engine (for text chat): retrieval + LLM synthesis
"""

from __future__ import annotations

from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from kbee.config import settings

SYSTEM_PROMPT = (
    "You are KBee, a helpful customer service AI assistant. "
    "Language policy: always reply in Traditional Chinese. "
    "Answer questions ONLY based on the provided knowledge base content. "
    "Keep replies concise. "
    "If the answer is not found in the knowledge base, say: "
    "'我的知識庫裡沒有這方面的資訊，請聯絡客服團隊取得進一步協助。'"
)

_INDEX_CACHE: VectorStoreIndex | None = None


def _get_index() -> VectorStoreIndex:
    """Return a cached VectorStoreIndex backed by ChromaDB."""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    if not settings.chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB storage not found at: {settings.chroma_path}. "
            "Run ingestion first: python -m kbee.ingest"
        )

    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(name=settings.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    _INDEX_CACHE = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return _INDEX_CACHE


def get_retriever() -> VectorIndexRetriever:
    """Return a retriever for fast vector-only search (no LLM)."""
    index = _get_index()
    return index.as_retriever(similarity_top_k=settings.similarity_top_k)


async def retrieve_chunks(query: str) -> list[dict[str, Any]]:
    """Retrieve relevant chunks from the knowledge base without LLM.

    Returns a list of dicts with text, score, and source metadata.
    """
    retriever = get_retriever()
    nodes = await retriever.aretrieve(query)
    results = []
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        results.append(
            {
                "text": getattr(node, "text", ""),
                "score": getattr(node, "score", None),
                "file_name": metadata.get("file_name", "Unknown"),
            }
        )
    return results


def get_query_engine() -> BaseQueryEngine:
    """Return a full query engine (retrieval + LLM synthesis)."""
    index = _get_index()
    llm = OpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        system_prompt=SYSTEM_PROMPT,
    )
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=settings.similarity_top_k,
    )


async def query_knowledge_base(
    query_engine: BaseQueryEngine,
    question: str,
) -> RESPONSE_TYPE:
    """Query the knowledge base with LLM synthesis."""
    return await query_engine.aquery(question)
