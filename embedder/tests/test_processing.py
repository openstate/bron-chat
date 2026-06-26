import pytest
from unittest.mock import MagicMock, patch

from app.processing import batch_generator, prepare_qdrant_payload, process_batch


class TestPrepareQdrantPayload:
    def _doc(self, source, description="content", extra=None):
        base = {
            "_id": "doc1",
            "_source": {
                "source": source,
                "description": description,
                "title": "Test",
                "location_name": "City",
                "doc_url": "",
                "location": "loc1",
                "modified": "2024-01-01",
                "published": "2024-01-01",
                "processed": "2024-06-01T10:00:00",
                "type": "test",
                "identifier": "id1",
                "url": "http://example.com",
            },
        }
        if extra:
            base["_source"].update(extra)
        return base

    def test_cvdr_uses_html_partition(self):
        with patch("app.processing.html_partition", return_value=["Chunk A", "Chunk B"]) as mock_part:
            payloads, processed = prepare_qdrant_payload(self._doc("cvdr"))
        mock_part.assert_called_once()
        assert payloads is not None
        assert len(payloads) == 2
        assert payloads[0]["meta"]["source"] == "cvdr"
        assert payloads[0]["meta"]["page_number"] == 0
        assert payloads[1]["meta"]["page_number"] == 1

    def test_oor_uses_txt_partition(self):
        with patch("app.processing.txt_partition", return_value=["Text chunk"]) as mock_part:
            payloads, processed = prepare_qdrant_payload(self._doc("oor"))
        mock_part.assert_called_once()
        assert payloads is not None

    def test_openbesluitvorming_with_md_text_uses_md_partition(self):
        doc = self._doc("openbesluitvorming", extra={"md_text": "# raadsvoorstel\n\n**veld** waarde"})
        with patch("app.processing.md_partition", return_value=["MD chunk"]) as mock_md, \
             patch("app.processing.html_txt_partition") as mock_html_txt:
            payloads, processed = prepare_qdrant_payload(doc)
        mock_md.assert_called_once()
        mock_html_txt.assert_not_called()
        assert payloads is not None

    def test_openbesluitvorming_without_md_text_falls_back_to_html_txt(self):
        with patch("app.processing.html_txt_partition", return_value=["Mixed"]) as mock_html_txt, \
             patch("app.processing.md_partition") as mock_md:
            payloads, processed = prepare_qdrant_payload(self._doc("openbesluitvorming"))
        mock_html_txt.assert_called_once()
        mock_md.assert_not_called()

    def test_unknown_source_defaults_to_txt(self):
        with patch("app.processing.txt_partition", return_value=["Default"]) as mock_part:
            payloads, processed = prepare_qdrant_payload(self._doc("unknown"))
        mock_part.assert_called_once()

    def test_returns_processed_timestamp(self):
        with patch("app.processing.html_partition", return_value=["Chunk"]):
            _, processed = prepare_qdrant_payload(self._doc("cvdr"))
        assert processed == "2024-06-01T10:00:00"

    def test_empty_chunks_returns_none(self):
        with patch("app.processing.html_partition", return_value=[]):
            payloads, processed = prepare_qdrant_payload(self._doc("cvdr"))
        assert payloads == []

    def test_partition_failure_returns_none(self):
        with patch("app.processing.html_partition", side_effect=Exception("fail")):
            payloads, processed = prepare_qdrant_payload(self._doc("cvdr"))
        assert payloads is None

    def test_missing_fields_default_to_empty(self):
        doc = {"_id": "doc1", "_source": {"source": "cvdr", "description": "x"}}
        with patch("app.processing.html_partition", return_value=["Chunk"]):
            payloads, _ = prepare_qdrant_payload(doc)
        assert payloads is not None
        assert payloads[0]["meta"]["title"] == ""
        assert payloads[0]["meta"]["location"] == ""

    def test_malformed_doc_returns_error(self):
        payloads, processed = prepare_qdrant_payload({"bad": "structure"})
        assert payloads is None

    def test_remove_processing_instructions_failure_continues(self):
        with patch("app.processing.remove_processing_instructions", side_effect=Exception("fail")), \
             patch("app.processing.html_partition", return_value=["Chunk A"]):
            payloads, processed = prepare_qdrant_payload(self._doc("cvdr"))
        assert payloads is not None  # processing still continued despite the error


class TestProcessBatch:
    def test_empty_payloads(self, mock_qdrant_client, mock_cohere_client, mock_sparse_embedder):
        upserted, errors = process_batch(mock_qdrant_client, mock_cohere_client, mock_sparse_embedder, [], "")
        assert upserted == 0
        assert errors == 0

    def test_all_existing_skips_cohere(self, mock_qdrant_client, mock_cohere_client, mock_sparse_embedder, sample_payloads):
        with patch("app.processing.filter_new_chunks", return_value=[]) as mock_filter, \
             patch("app.processing.write_last_run") as mock_state:
            upserted, errors = process_batch(
                mock_qdrant_client, mock_cohere_client, mock_sparse_embedder,
                sample_payloads, "2024-06-01T10:00:00"
            )
        mock_filter.assert_called_once()
        mock_cohere_client.embed.assert_not_called()
        mock_state.assert_called_once_with("2024-06-01T10:00:00")
        assert upserted == 0

    def test_success_advances_state(self, mock_qdrant_client, mock_cohere_client, mock_sparse_embedder, sample_payloads):
        pairs = [("id1", sample_payloads[0])]
        with patch("app.processing.filter_new_chunks", return_value=pairs), \
             patch("app.processing.generate_dense_embeddings", return_value=[[0.1, 0.2]]), \
             patch("app.processing.generate_sparse_embedding", return_value=[MagicMock()]), \
             patch("app.processing.make_qdrant_points", return_value=[MagicMock()]), \
             patch("app.processing.upsert_points_to_qdrant"), \
             patch("app.processing.write_last_run") as mock_state:
            upserted, errors = process_batch(
                mock_qdrant_client, mock_cohere_client, mock_sparse_embedder,
                sample_payloads, "2024-06-01T10:00:00"
            )
        mock_state.assert_called_once_with("2024-06-01T10:00:00")
        assert upserted == 1
        assert errors == 0

    def test_dense_embedding_failure_no_state_advance(self, mock_qdrant_client, mock_cohere_client, mock_sparse_embedder, sample_payloads):
        pairs = [("id1", sample_payloads[0])]
        with patch("app.processing.filter_new_chunks", return_value=pairs), \
             patch("app.processing.generate_dense_embeddings", return_value=None), \
             patch("app.processing.write_last_run") as mock_state:
            upserted, errors = process_batch(
                mock_qdrant_client, mock_cohere_client, mock_sparse_embedder,
                sample_payloads, "2024-06-01T10:00:00"
            )
        mock_state.assert_not_called()
        assert upserted == 0
        assert errors > 0

    def test_upsert_failure_no_state_advance(self, mock_qdrant_client, mock_cohere_client, mock_sparse_embedder, sample_payloads):
        pairs = [("id1", sample_payloads[0])]
        with patch("app.processing.filter_new_chunks", return_value=pairs), \
             patch("app.processing.generate_dense_embeddings", return_value=[[0.1]]), \
             patch("app.processing.generate_sparse_embedding", return_value=[MagicMock()]), \
             patch("app.processing.make_qdrant_points", return_value=[MagicMock()]), \
             patch("app.processing.upsert_points_to_qdrant", side_effect=Exception("fail")), \
             patch("app.processing.write_last_run") as mock_state:
            upserted, errors = process_batch(
                mock_qdrant_client, mock_cohere_client, mock_sparse_embedder,
                sample_payloads, "2024-06-01T10:00:00"
            )
        mock_state.assert_not_called()
        assert upserted == 0
        assert errors > 0


class TestBatchGenerator:
    def _settings(self, batch_size=5, threshold_mult=2, max_docs=-1, index="test", scroll="5m"):
        m = MagicMock()
        m.BATCH_SIZE = batch_size
        m.PROCESSING_THRESHOLD_MULTIPLIER = threshold_mult
        m.MAX_DOCUMENTS_TO_PROCESS = max_docs
        m.ES_INDEX = index
        m.ES_SCROLL_TIME = scroll
        return m

    def _fake_doc(self, doc_id="doc1"):
        return {"_id": doc_id, "_source": {}}

    def _payloads(self, source_id="doc1"):
        return [{"content": "c", "meta": {"source_id": source_id}}]

    def test_empty_scan_yields_nothing(self):
        with patch("app.processing.scan", return_value=iter([])), \
             patch("app.processing.settings", self._settings()):
            result = list(batch_generator(MagicMock(), {}))
        assert result == []

    def test_yields_remaining_batch_in_finally(self):
        docs = [self._fake_doc("doc1")]
        with patch("app.processing.scan", return_value=iter(docs)), \
             patch("app.processing.settings", self._settings()), \
             patch("app.processing.prepare_qdrant_payload", return_value=(self._payloads(), "2024-01-01")):
            result = list(batch_generator(MagicMock(), {}, max_documents=-1))
        assert len(result) == 1
        batch, ts = result[0]
        assert batch == self._payloads()
        assert ts == "2024-01-01"

    def test_yields_mid_scan_when_threshold_reached(self):
        # threshold = batch_size(1) * multiplier(1) = 1, so each doc triggers a yield
        docs = [self._fake_doc("doc1"), self._fake_doc("doc2")]
        with patch("app.processing.scan", return_value=iter(docs)), \
             patch("app.processing.settings", self._settings(batch_size=1, threshold_mult=1)), \
             patch("app.processing.prepare_qdrant_payload", return_value=(self._payloads(), "2024-01-01")):
            result = list(batch_generator(MagicMock(), {}, max_documents=-1))
        assert len(result) == 2

    def test_respects_max_documents_limit(self):
        docs = [self._fake_doc(f"doc{i}") for i in range(5)]
        with patch("app.processing.scan", return_value=iter(docs)), \
             patch("app.processing.settings", self._settings()), \
             patch("app.processing.prepare_qdrant_payload", return_value=(self._payloads(), "2024-01-01")):
            result = list(batch_generator(MagicMock(), {}, max_documents=2))
        batch, _ = result[0]
        assert len(batch) == 2

    def test_max_documents_minus_one_processes_all(self):
        docs = [self._fake_doc(f"doc{i}") for i in range(10)]
        with patch("app.processing.scan", return_value=iter(docs)), \
             patch("app.processing.settings", self._settings()), \
             patch("app.processing.prepare_qdrant_payload", return_value=(self._payloads(), "2024-01-01")):
            result = list(batch_generator(MagicMock(), {}, max_documents=-1))
        batch, _ = result[0]
        assert len(batch) == 10

    def test_scan_exception_yields_partial_batch_then_raises(self):
        def failing_scan(**kwargs):
            yield self._fake_doc("doc1")
            raise Exception("scroll timeout")

        with patch("app.processing.scan", side_effect=failing_scan), \
             patch("app.processing.settings", self._settings()), \
             patch("app.processing.prepare_qdrant_payload", return_value=(self._payloads(), "2024-01-01")):
            generator = batch_generator(MagicMock(), {}, max_documents=-1)
            batch, _ = next(generator)  # partial batch still yielded for upsert
            with pytest.raises(RuntimeError, match="scan failed mid-run"):
                next(generator)  # then the run fails loudly instead of ending clean
        assert batch == self._payloads()

    def test_scan_sorts_by_processed_and_preserves_order(self):
        with patch("app.processing.scan", return_value=iter([])) as mock_scan, \
             patch("app.processing.settings", self._settings()):
            list(batch_generator(MagicMock(), {"query": {"match_all": {}}}))
        kwargs = mock_scan.call_args.kwargs
        # Sorted ascending so batch_max_processed is a true resume watermark.
        assert kwargs["query"]["sort"] == [{"processed": {"order": "asc"}}]
        assert kwargs["query"]["query"] == {"match_all": {}}
        assert kwargs["preserve_order"] is True

    def test_docs_with_no_payloads_are_skipped(self):
        docs = [self._fake_doc("doc1")]
        with patch("app.processing.scan", return_value=iter(docs)), \
             patch("app.processing.settings", self._settings()), \
             patch("app.processing.prepare_qdrant_payload", return_value=(None, "2024-01-01")):
            result = list(batch_generator(MagicMock(), {}, max_documents=-1))
        assert result == []

    def test_tracks_max_processed_timestamp(self):
        docs = [self._fake_doc("doc1"), self._fake_doc("doc2")]
        prepare_returns = [
            (self._payloads("doc1"), "2024-01-02T00:00:00"),
            (self._payloads("doc2"), "2024-01-01T00:00:00"),  # earlier — should not win
        ]
        with patch("app.processing.scan", return_value=iter(docs)), \
             patch("app.processing.settings", self._settings()), \
             patch("app.processing.prepare_qdrant_payload", side_effect=prepare_returns):
            result = list(batch_generator(MagicMock(), {}, max_documents=-1))
        _, ts = result[0]
        assert ts == "2024-01-02T00:00:00"

    def test_uses_settings_max_documents_when_not_passed(self):
        docs = [self._fake_doc(f"doc{i}") for i in range(5)]
        with patch("app.processing.scan", return_value=iter(docs)), \
             patch("app.processing.settings", self._settings(max_docs=2)), \
             patch("app.processing.prepare_qdrant_payload", return_value=(self._payloads(), "2024-01-01")):
            result = list(batch_generator(MagicMock(), {}))
        batch, _ = result[0]
        assert len(batch) == 2


class TestProcessBatchMissingPaths:
    def test_qdrant_unreachable_raises(self, mock_qdrant_client, mock_cohere_client, mock_sparse_embedder, sample_payloads):
        with patch("app.processing.filter_new_chunks", side_effect=Exception("Connection error")):
            with pytest.raises(Exception, match="Connection error"):
                process_batch(
                    mock_qdrant_client, mock_cohere_client, mock_sparse_embedder,
                    sample_payloads, "2024-06-01T10:00:00",
                )

    def test_sparse_embedding_failure_returns_errors(
        self, mock_qdrant_client, mock_cohere_client, mock_sparse_embedder, sample_payloads
    ):
        pairs = [("id1", sample_payloads[0])]
        with patch("app.processing.filter_new_chunks", return_value=pairs), \
             patch("app.processing.generate_dense_embeddings", return_value=[[0.1]]), \
             patch("app.processing.generate_sparse_embedding", return_value=None), \
             patch("app.processing.write_last_run") as mock_state:
            upserted, errors = process_batch(
                mock_qdrant_client, mock_cohere_client, mock_sparse_embedder,
                sample_payloads, "2024-06-01T10:00:00",
            )
        mock_state.assert_not_called()
        assert upserted == 0
        assert errors == len(pairs)

    def test_empty_qdrant_points_returns_errors(
        self, mock_qdrant_client, mock_cohere_client, mock_sparse_embedder, sample_payloads
    ):
        pairs = [("id1", sample_payloads[0])]
        with patch("app.processing.filter_new_chunks", return_value=pairs), \
             patch("app.processing.generate_dense_embeddings", return_value=[[0.1]]), \
             patch("app.processing.generate_sparse_embedding", return_value=[MagicMock()]), \
             patch("app.processing.make_qdrant_points", return_value=[]), \
             patch("app.processing.write_last_run") as mock_state:
            upserted, errors = process_batch(
                mock_qdrant_client, mock_cohere_client, mock_sparse_embedder,
                sample_payloads, "2024-06-01T10:00:00",
            )
        mock_state.assert_not_called()
        assert upserted == 0
        assert errors == len(pairs)
