import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import settings
from app.estimate import (
    MISSING_SOURCE,
    SPARSE_BYTES_PER_ELEMENT,
    FALLBACK_DENSE_BYTES,
    bootstrap_ci,
    count_billable_tokens,
    estimate_source,
    extrapolate,
    get_dense_vector_bytes,
    get_source_counts,
    point_bytes,
    run_estimate,
    sample_docs,
)


class FakeTokenizer:
    """One token per whitespace-separated word."""

    def encode(self, text):
        return SimpleNamespace(ids=list(range(len(text.split()))))


def _es_doc(doc_id, source="cvdr", description="some text"):
    return {
        "_id": doc_id,
        "_source": {
            "source": source,
            "description": description,
            "title": "Test",
            "processed": "2024-06-01T10:00:00",
        },
    }


def _payload(source_id, page_number, content):
    return {
        "content": content,
        "meta": {"source_id": source_id, "page_number": page_number, "source": "cvdr"},
    }


class TestCountBillableTokens:
    def test_counts_tokens_below_cap(self):
        assert count_billable_tokens(FakeTokenizer(), "one two three") == 3

    def test_caps_at_cohere_max_tokens(self):
        text = "word " * (settings.COHERE_MAX_TOKENS + 100)
        assert count_billable_tokens(FakeTokenizer(), text) == settings.COHERE_MAX_TOKENS

    def test_heuristic_fallback_without_tokenizer(self):
        text = "x" * 35
        assert count_billable_tokens(None, text) == math.ceil(
            35 / settings.ESTIMATE_CHARS_PER_TOKEN
        )


class TestPointBytes:
    def test_sums_dense_sparse_and_payload(self):
        payload = _payload("doc1", 0, "hello world")
        expected_payload_bytes = len(json.dumps(payload).encode("utf-8"))
        assert point_bytes(payload, 10, 1024) == (
            1024 + 10 * SPARSE_BYTES_PER_ELEMENT + expected_payload_bytes
        )


class TestGetDenseVectorBytes:
    def test_fallback_without_qdrant(self):
        size, description = get_dense_vector_bytes(None)
        assert size == FALLBACK_DENSE_BYTES
        assert "assumed" in description

    def test_reads_dims_and_datatype_from_collection(self):
        client = MagicMock()
        client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={
                        "text-dense": SimpleNamespace(
                            size=1024, datatype=SimpleNamespace(value="uint8")
                        )
                    }
                )
            )
        )
        size, description = get_dense_vector_bytes(client)
        assert size == 1024
        assert "from collection" in description

    def test_missing_datatype_defaults_to_float32(self):
        client = MagicMock()
        client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"text-dense": SimpleNamespace(size=1024, datatype=None)}
                )
            )
        )
        size, _ = get_dense_vector_bytes(client)
        assert size == 1024 * 4

    def test_unreadable_collection_falls_back(self):
        client = MagicMock()
        client.get_collection.side_effect = Exception("boom")
        size, description = get_dense_vector_bytes(client)
        assert size == FALLBACK_DENSE_BYTES
        assert "assumed" in description


class TestExtrapolate:
    def test_mean_times_population(self):
        assert extrapolate([10, 20, 30], 100) == 2000

    def test_empty_sample_is_zero(self):
        assert extrapolate([], 100) == 0.0


class TestBootstrapCi:
    def test_deterministic_under_fixed_seed(self):
        values = list(range(100))
        first = bootstrap_ci([(values, 1000)], 200, seed=42)
        second = bootstrap_ci([(values, 1000)], 200, seed=42)
        assert first == second

    def test_brackets_point_estimate(self):
        values = [10] * 50 + [30] * 50
        low, high = bootstrap_ci([(values, 1000)], 500, seed=42)
        point = extrapolate(values, 1000)
        assert low <= point <= high
        assert low < high

    def test_constant_sample_collapses_ci(self):
        low, high = bootstrap_ci([([5, 5, 5], 100)], 100, seed=1)
        assert low == high == 500

    def test_empty_strata(self):
        assert bootstrap_ci([([], 100)], 100, seed=1) == (0.0, 0.0)

    def test_pooled_strata_sum(self):
        low, high = bootstrap_ci([([5, 5], 100), ([10, 10], 10)], 100, seed=1)
        assert low == high == 500 + 100


class TestQueryConstruction:
    BASE_QUERY = {"range": {"processed": {"gte": "2026-01-01T00:00:00"}}}

    def test_source_counts_aggregation(self, mock_es_client):
        mock_es_client.search.return_value = {
            "aggregations": {
                "by_source": {
                    "buckets": [
                        {"key": "cvdr", "doc_count": 10},
                        {"key": "oor", "doc_count": 0},
                    ]
                }
            }
        }
        counts = get_source_counts(mock_es_client, self.BASE_QUERY)

        assert counts == {"cvdr": 10}
        kwargs = mock_es_client.search.call_args.kwargs
        assert kwargs["size"] == 0
        assert kwargs["query"] == self.BASE_QUERY
        terms = kwargs["aggs"]["by_source"]["terms"]
        assert terms["field"] == settings.ESTIMATE_SOURCE_FIELD
        assert terms["missing"] == MISSING_SOURCE

    def test_sample_docs_random_score_and_filters(self, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        sample_docs(mock_es_client, "cvdr", self.BASE_QUERY, n=50, seed=7)

        kwargs = mock_es_client.search.call_args.kwargs
        assert kwargs["size"] == 50
        function_score = kwargs["query"]["function_score"]
        assert function_score["random_score"] == {"seed": 7, "field": "_seq_no"}
        filters = function_score["query"]["bool"]["filter"]
        assert self.BASE_QUERY in filters
        assert {"term": {settings.ESTIMATE_SOURCE_FIELD: "cvdr"}} in filters

    def test_sample_docs_missing_source_bucket(self, mock_es_client):
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        sample_docs(mock_es_client, MISSING_SOURCE, self.BASE_QUERY, n=5, seed=7)

        filters = mock_es_client.search.call_args.kwargs["query"]["function_score"][
            "query"
        ]["bool"]["filter"]
        assert {
            "bool": {"must_not": {"exists": {"field": settings.ESTIMATE_SOURCE_FIELD}}}
        } in filters

    def test_sample_size_capped_at_population(self, mock_es_client):
        mock_es_client.search.side_effect = [
            {"hits": {"hits": []}},
        ]
        with patch("app.estimate.prepare_qdrant_payload"):
            estimate_source(
                mock_es_client, None, None, FakeTokenizer(),
                "cvdr", 3, self.BASE_QUERY, sample_size=500, seed=1, dense_bytes=0,
            )
        assert mock_es_client.search.call_args.kwargs["size"] == 3


class TestEstimateSource:
    BASE_QUERY = {"range": {"processed": {"gte": "1970-01-01T00:00:00"}}}

    def _run(self, mock_es_client, qdrant_client, prepare_side_effect, new_pairs):
        mock_es_client.search.return_value = {
            "hits": {"hits": [_es_doc("doc1"), _es_doc("doc2"), _es_doc("doc3")]}
        }
        with patch(
            "app.estimate.prepare_qdrant_payload", side_effect=prepare_side_effect
        ), patch("app.estimate.filter_new_chunks", return_value=new_pairs):
            return estimate_source(
                mock_es_client, qdrant_client, None, FakeTokenizer(),
                "cvdr", 300, self.BASE_QUERY, sample_size=3, seed=1, dense_bytes=1024,
            )

    def test_per_doc_zeros_kept_after_dedup_and_failures(self, mock_es_client, mock_qdrant_client):
        chunk1 = _payload("doc1", 0, "word " * 10)
        chunk2 = _payload("doc1", 1, "word " * 20)
        chunk3 = _payload("doc2", 0, "word " * 5)

        def prepare(doc):
            return {
                "doc1": ([chunk1, chunk2], "2024-01-01"),
                "doc2": ([chunk3], "2024-01-01"),  # fully deduped below
                "doc3": (None, None),  # partition failure
            }[doc["_id"]]

        # Dedup keeps only doc1's chunks.
        new_pairs = [("id-a", chunk1), ("id-b", chunk2)]
        est = self._run(mock_es_client, mock_qdrant_client, prepare, new_pairs)

        assert est.n_sampled == 3
        assert est.chunk_count == 3
        assert est.new_chunk_count == 2
        assert est.partition_failures == 1
        # One entry per sampled doc; deduped and failed docs stay as zeros.
        assert sorted(est.per_doc_tokens) == [0, 0, 30]
        assert len(est.per_doc_bytes) == 3
        assert sorted(est.per_doc_bytes)[:2] == [0, 0]
        assert est.est_tokens == 10 * 300  # mean(30, 0, 0) * 300

    def test_no_dedup_counts_all_chunks(self, mock_es_client):
        chunk1 = _payload("doc1", 0, "word " * 10)

        def prepare(doc):
            return ([chunk1], "2024-01-01") if doc["_id"] == "doc1" else ([], "2024-01-01")

        with patch("app.estimate.filter_new_chunks") as mock_filter:
            mock_es_client.search.return_value = {
                "hits": {"hits": [_es_doc("doc1"), _es_doc("doc2")]}
            }
            with patch("app.estimate.prepare_qdrant_payload", side_effect=prepare):
                est = estimate_source(
                    mock_es_client, None, None, FakeTokenizer(),
                    "cvdr", 100, self.BASE_QUERY, sample_size=2, seed=1, dense_bytes=0,
                )
        mock_filter.assert_not_called()
        assert est.new_chunk_count == 1
        assert sorted(est.per_doc_tokens) == [0, 10]


class TestRunEstimate:
    def test_smoke_read_only(self, mock_es_client, mock_qdrant_client):
        mock_cohere_client = MagicMock()
        # models.get returns a MagicMock whose tokenizer_url is not a str -> heuristic.
        agg_response = {
            "aggregations": {
                "by_source": {"buckets": [{"key": "cvdr", "doc_count": 100}]}
            }
        }
        sample_response = {
            "hits": {
                "hits": [
                    _es_doc("doc1", description="Eerste alinea over het besluit. " * 5),
                    _es_doc("doc2", description="Tweede alinea over de regeling. " * 5),
                ]
            }
        }
        mock_es_client.search.side_effect = [agg_response, sample_response]

        with patch(
            "app.processing.html_partition",
            return_value=["Chunk over het besluit", "Tweede chunk"],
        ), patch("app.state.write_last_run") as mock_state:
            estimates = run_estimate(
                mock_es_client,
                mock_qdrant_client,
                mock_cohere_client,
                MagicMock(),  # sparse embedder; iteration yields nothing -> fallback
                {"range": {"processed": {"gte": "1970-01-01T00:00:00"}}},
                sample_size=2,
                seed=42,
            )

        assert len(estimates) == 1
        assert estimates[0].n_total == 100
        assert estimates[0].n_sampled == 2
        assert estimates[0].chunk_count == 4
        assert estimates[0].est_tokens > 0

        # Read-only guarantees: nothing embedded, upserted, or persisted.
        mock_cohere_client.embed.assert_not_called()
        mock_qdrant_client.upsert.assert_not_called()
        mock_state.assert_not_called()

    def test_no_documents_in_scope(self, mock_es_client):
        mock_es_client.search.return_value = {
            "aggregations": {"by_source": {"buckets": []}}
        }
        estimates = run_estimate(
            mock_es_client, None, None, None,
            {"range": {"processed": {"gte": "1970-01-01T00:00:00"}}},
            sample_size=10,
            seed=42,
        )
        assert estimates == []
