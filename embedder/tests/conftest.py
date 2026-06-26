import logging
import pytest
from unittest.mock import MagicMock

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def mock_es_client():
    client = MagicMock()
    client.indices.exists = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.get_collection = MagicMock(return_value={"name": "test_collection"})
    client.retrieve = MagicMock(return_value=[])
    return client


@pytest.fixture
def mock_cohere_client():
    return MagicMock()


@pytest.fixture
def mock_sparse_embedder():
    return MagicMock()


@pytest.fixture
def sample_payloads():
    return [
        {
            "content": "text1",
            "meta": {
                "source_id": "doc1",
                "page_number": 0,
                "page_count": 1,
                "title": "Doc 1",
                "location": "loc1",
                "location_name": "Location 1",
                "modified": "2024-01-01",
                "published": "2024-01-01",
                "type": "test",
                "identifier": "id1",
                "url": "http://example.com/1",
                "doc_url": "",
                "source": "cvdr",
            },
        },
        {
            "content": "text2",
            "meta": {
                "source_id": "doc2",
                "page_number": 0,
                "page_count": 1,
                "title": "Doc 2",
                "location": "loc2",
                "location_name": "Location 2",
                "modified": "2024-01-02",
                "published": "2024-01-02",
                "type": "test",
                "identifier": "id2",
                "url": "http://example.com/2",
                "doc_url": "",
                "source": "oor",
            },
        },
    ]
