"""Basic tests for KBee ingestion and configuration."""

import pytest

from kbee.config import Settings, settings


class TestConfig:
    """Tests for application configuration."""

    def test_settings_instance_exists(self) -> None:
        """Settings singleton should be importable."""
        assert settings is not None

    def test_settings_defaults(self) -> None:
        """Default settings should have expected values."""
        s = Settings(openai_api_key="test-key")
        assert s.llm_model == "gpt-5-mini"
        assert s.embedding_model == "text-embedding-3-small"
        assert s.chunk_size == 1024
        assert s.chunk_overlap == 200
        assert s.similarity_top_k == 5
        assert s.collection_name == "kbee_docs"

    def test_chroma_path_property(self) -> None:
        """chroma_path should return a Path object."""
        s = Settings(openai_api_key="test-key", chroma_persist_dir="/tmp/test")
        assert str(s.chroma_path) == "/tmp/test"

    def test_data_path_property(self) -> None:
        """data_path should return a Path object."""
        s = Settings(openai_api_key="test-key", data_dir="/tmp/data")
        assert str(s.data_path) == "/tmp/data"


class TestIngestImports:
    """Tests for ingestion module imports."""

    def test_ingest_module_importable(self) -> None:
        """Ingestion module should be importable."""
        from kbee.ingest import ingest_documents  # noqa: F401

    def test_ingest_no_dir_raises(self) -> None:
        """Ingestion should raise FileNotFoundError for missing dir."""
        from kbee.ingest import ingest_documents

        with pytest.raises(FileNotFoundError):
            ingest_documents(data_dir="/nonexistent/path")


class TestQueryImports:
    """Tests for query module imports."""

    def test_query_module_importable(self) -> None:
        """Query module should be importable."""
        from kbee.query import get_query_engine  # noqa: F401

    def test_query_function_importable(self) -> None:
        """Async query function should be importable."""
        from kbee.query import query_knowledge_base  # noqa: F401
