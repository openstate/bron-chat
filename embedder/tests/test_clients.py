from unittest.mock import MagicMock, patch

import pytest

from app.clients import (
    get_cohere_client,
    get_elasticsearch_client,
    get_qdrant_client,
    get_sparse_embedder,
)

PATCH = "app.clients.{}"


class TestGetElasticsearchClient:
    def test_returns_client_when_index_exists(self):
        mock_es = MagicMock()
        mock_es.indices.exists.return_value = True
        with patch(PATCH.format("Elasticsearch"), return_value=mock_es):
            client = get_elasticsearch_client()
        assert client is mock_es

    def test_raises_when_index_missing(self):
        mock_es = MagicMock()
        mock_es.indices.exists.return_value = False
        with patch(PATCH.format("Elasticsearch"), return_value=mock_es):
            with pytest.raises(ValueError, match="does not exist"):
                get_elasticsearch_client()


class TestGetQdrantClient:
    def test_returns_client_when_collection_exists(self):
        mock_qdrant = MagicMock()
        with patch(PATCH.format("QdrantClient"), return_value=mock_qdrant):
            client = get_qdrant_client()
        assert client is mock_qdrant

    def test_raises_when_collection_missing(self):
        mock_qdrant = MagicMock()
        mock_qdrant.get_collection.side_effect = Exception("not found")
        with patch(PATCH.format("QdrantClient"), return_value=mock_qdrant):
            with pytest.raises(ValueError, match="does not exist"):
                get_qdrant_client()


class TestGetCohereClient:
    def test_returns_client_when_key_set(self):
        with patch(PATCH.format("settings")) as mock_settings, \
             patch(PATCH.format("cohere")) as mock_cohere:
            mock_settings.COHERE_API_KEY = "test-key"
            get_cohere_client()
        mock_cohere.ClientV2.assert_called_once_with(api_key="test-key")

    def test_raises_when_key_missing(self):
        with patch(PATCH.format("settings")) as mock_settings:
            mock_settings.COHERE_API_KEY = ""
            with pytest.raises(ValueError, match="COHERE_API_KEY"):
                get_cohere_client()


class TestGetSparseEmbedder:
    def test_returns_embedder(self):
        with patch(PATCH.format("ort")), \
             patch(PATCH.format("SparseTextEmbedding")) as mock_cls, \
             patch(PATCH.format("settings")) as mock_settings:
            mock_settings.NUM_WORKERS = 4
            mock_settings.SPARSE_MODEL_NAME = "prithvida/Splade_PP_en_v1"
            mock_settings.MODELS_DIR = "/models"
            mock_settings.SPARSE_PROVIDERS = ["CPUExecutionProvider"]
            result = get_sparse_embedder()
        mock_cls.assert_called_once()
        assert result is mock_cls.return_value
