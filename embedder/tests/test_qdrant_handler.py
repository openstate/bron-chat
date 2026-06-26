import uuid
import pytest
import numpy as np
from typing import Any, List
from unittest.mock import MagicMock, patch
from qdrant_client.models import PointStruct
from fastembed.sparse import SparseEmbedding

from app.qdrant_handler import _chunk_id, filter_new_chunks, make_qdrant_points, upsert_points_to_qdrant


class TestChunkId:
    def test_different_for_different_inputs(self):
        assert _chunk_id("doc1", 0) != _chunk_id("doc1", 1)
        assert _chunk_id("doc1", 0) != _chunk_id("doc2", 0)

    def test_is_valid_uuid(self):
        cid = _chunk_id("some-source-id", 3)
        uuid.UUID(cid)  # raises if invalid


class TestFilterNewChunks:
    def test_all_new(self, mock_qdrant_client, sample_payloads):
        mock_qdrant_client.retrieve.return_value = []
        result = filter_new_chunks(mock_qdrant_client, sample_payloads)
        assert len(result) == len(sample_payloads)

    def test_all_existing(self, mock_qdrant_client, sample_payloads):
        expected_ids = [_chunk_id(p["meta"]["source_id"], p["meta"]["page_number"]) for p in sample_payloads]
        existing = [MagicMock(id=cid) for cid in expected_ids]
        mock_qdrant_client.retrieve.return_value = existing
        result = filter_new_chunks(mock_qdrant_client, sample_payloads)
        assert len(result) == 0

    def test_partial_existing(self, mock_qdrant_client, sample_payloads):
        first_id = _chunk_id(sample_payloads[0]["meta"]["source_id"], sample_payloads[0]["meta"]["page_number"])
        mock_qdrant_client.retrieve.return_value = [MagicMock(id=first_id)]
        result = filter_new_chunks(mock_qdrant_client, sample_payloads)
        assert len(result) == 1
        assert result[0][1]["meta"]["source_id"] == sample_payloads[1]["meta"]["source_id"]

    def test_qdrant_error_raises(self, mock_qdrant_client, sample_payloads):
        mock_qdrant_client.retrieve.side_effect = Exception("Connection error")
        with pytest.raises(Exception, match="Connection error"):
            filter_new_chunks(mock_qdrant_client, sample_payloads)


class TestMakeQdrantPoints:
    def _make_sparse(self, indices, values):
        s = MagicMock(spec=SparseEmbedding)
        s.indices = np.array(indices)
        s.values = np.array(values)
        return s

    def test_success(self, sample_payloads):
        pairs = [(_chunk_id(p["meta"]["source_id"], p["meta"]["page_number"]), p) for p in sample_payloads]
        dense = [[1, 2, 3], [4, 5, 6]]
        sparse: List[Any] = [self._make_sparse([0, 1], [0.5, 0.8]), self._make_sparse([1, 2], [0.3, 0.9])]

        points = make_qdrant_points(pairs, dense, sparse)

        assert len(points) == 2
        assert points[0].id == pairs[0][0]
        assert points[1].id == pairs[1][0]

    def test_mismatched_lengths(self, sample_payloads):
        pairs = [(_chunk_id(p["meta"]["source_id"], p["meta"]["page_number"]), p) for p in sample_payloads]
        result = make_qdrant_points(pairs, [[1, 2]], [])
        assert result == []

    def test_invalid_sparse_format(self, sample_payloads):
        pairs = [(_chunk_id(sample_payloads[0]["meta"]["source_id"], 0), sample_payloads[0])]
        invalid_sparse: List[Any] = ["invalid"]
        result = make_qdrant_points(pairs, [[1, 2, 3]], invalid_sparse)
        assert len(result) == 0

    def test_empty_inputs(self):
        assert make_qdrant_points([], [], []) == []


class TestUpsertPointsToQdrant:
    def test_success(self, mock_qdrant_client):
        points: List[Any] = [MagicMock(spec=PointStruct), MagicMock(spec=PointStruct)]
        with patch("app.qdrant_handler.tqdm") as mock_tqdm:
            mock_tqdm.return_value.__enter__.return_value = MagicMock()
            upsert_points_to_qdrant(mock_qdrant_client, points, batch_size=1)
        assert mock_qdrant_client.upsert.call_count == 2

    def test_empty_points(self, mock_qdrant_client):
        upsert_points_to_qdrant(mock_qdrant_client, [])
        mock_qdrant_client.upsert.assert_not_called()

    def test_upsert_failure_raises(self, mock_qdrant_client):
        mock_qdrant_client.upsert.side_effect = Exception("upsert failed")
        with pytest.raises(Exception, match="upsert failed"):
            upsert_points_to_qdrant(mock_qdrant_client, [MagicMock(spec=PointStruct)])

    def test_custom_collection(self, mock_qdrant_client):
        points: List[Any] = [MagicMock(spec=PointStruct)]
        with patch("app.qdrant_handler.tqdm") as mock_tqdm:
            mock_tqdm.return_value.__enter__.return_value = MagicMock()
            upsert_points_to_qdrant(mock_qdrant_client, points, collection_name="custom")
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "custom"
