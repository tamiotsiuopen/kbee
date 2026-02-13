"""Document ingestion pipeline for KBee.

Reads documents from a directory, chunks them, generates embeddings,
and stores them in ChromaDB for later retrieval.
"""

import argparse
import logging
import sys
from pathlib import Path

from typing import Optional
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from kbee.config import settings

# Lazy-loaded chroma client holder
_CHROMA_CLIENT = None


def get_chroma_client(clear: bool = False):
    """Lazily create and return a Chroma persistent client and collection.

    Returns a tuple (client, collection).
    """
    global _CHROMA_CLIENT
    # Import here to avoid top-level dependency during module import
    import chromadb

    if _CHROMA_CLIENT is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(settings.chroma_path))

    client = _CHROMA_CLIENT
    if clear:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=settings.collection_name)
    return client, collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_chroma_collection(
    clear: bool = False,
) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    """Create or retrieve the ChromaDB collection.

    Args:
        clear: If True, delete and recreate the collection.

    Returns:
        A tuple of (client, collection).
    """
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))

    if clear:
        try:
            client.delete_collection(settings.collection_name)
            logger.info("Cleared existing collection: %s", settings.collection_name)
        except ValueError:
            pass  # Collection doesn't exist yet.

    collection = client.get_or_create_collection(name=settings.collection_name)
    return client, collection


def ingest_documents(data_dir: str | None = None, clear: bool = False) -> int:
    """Ingest documents from a directory into ChromaDB.

    Args:
        data_dir: Path to the directory containing documents.
            Defaults to settings.data_dir.
        clear: If True, clear existing data before ingesting.

    Returns:
        The number of document chunks ingested.

    Raises:
        FileNotFoundError: If the data directory does not exist.
        ValueError: If no documents are found in the directory.
    """
    dir_path = Path(data_dir) if data_dir else settings.data_path
    if not dir_path.exists():
        raise FileNotFoundError(f"Data directory not found: {dir_path}")

    # Read documents.
    logger.info("Reading documents from: %s", dir_path)
    reader = SimpleDirectoryReader(
        input_dir=str(dir_path),
        recursive=True,
        required_exts=[".pdf", ".txt", ".docx", ".md", ".html"],
    )

    try:
        documents = reader.load_data()
    except Exception as e:
        logger.error("Failed to read documents: %s", e)
        raise

    if not documents:
        raise ValueError(f"No documents found in: {dir_path}")

    logger.info("Loaded %d documents.", len(documents))

    # Set up embedding model.
    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    # Set up ChromaDB vector store.
    # Lazy get client/collection
    _, collection = get_chroma_client(clear=clear)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Parse and index documents.
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    logger.info(
        "Indexing with chunk_size=%d, overlap=%d...",
        settings.chunk_size,
        settings.chunk_overlap,
    )
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    num_nodes = len(index.docstore.docs)
    logger.info(
        "Ingested %d chunks into collection '%s'.",
        num_nodes,
        settings.collection_name,
    )
    return num_nodes


def main() -> None:
    """CLI entry point for document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into KBee knowledge base.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=f"Directory containing documents (default: {settings.data_dir})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingesting.",
    )
    args = parser.parse_args()

    try:
        count = ingest_documents(data_dir=args.dir, clear=args.clear)
        logger.info("Done! %d chunks ready for queries.", count)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Ingestion failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
