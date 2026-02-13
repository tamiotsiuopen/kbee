"""Application configuration via environment variables.

Uses pydantic-settings to load configuration from .env file
and environment variables.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        openai_api_key: OpenAI API key for LLM and embedding calls.
        chroma_persist_dir: Directory for ChromaDB persistent storage.
        data_dir: Directory containing knowledge base documents.
        llm_model: OpenAI model name for generating responses.
        embedding_model: OpenAI model name for text embeddings.
        chunk_size: Number of characters per text chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        similarity_top_k: Number of similar documents to retrieve.
        collection_name: ChromaDB collection name.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    chroma_persist_dir: str = "./storage"
    data_dir: str = "./data"
    llm_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1024
    chunk_overlap: int = 200
    similarity_top_k: int = 5
    collection_name: str = "kbee_docs"

    @property
    def chroma_path(self) -> Path:
        """Return the ChromaDB storage path as a Path object."""
        return Path(self.chroma_persist_dir)

    @property
    def data_path(self) -> Path:
        """Return the data directory path as a Path object."""
        return Path(self.data_dir)


settings = Settings()
