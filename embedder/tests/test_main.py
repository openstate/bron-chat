from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.state import EPOCH
from run import format_progress, main


@pytest.fixture
def patched_main():
    """Patch the four client constructors main() invokes at startup.
    Tests still patch read_last_run / batch_generator / process_batch as needed."""
    with patch("run.get_elasticsearch_client") as es, patch("run.get_qdrant_client") as qdrant, patch(
        "run.get_cohere_client"
    ) as cohere, patch("run.get_sparse_embedder") as sparse:
        yield SimpleNamespace(es=es, qdrant=qdrant, cohere=cohere, sparse=sparse)


class TestMain:
    def test_no_batches_runs_without_error(self, patched_main):
        with patch("run.read_last_run", return_value="2024-01-01T00:00:00"), patch(
            "run.batch_generator", return_value=iter([])
        ), patch("run.process_batch") as mock_process:
            main([])
        mock_process.assert_not_called()

    def test_empty_batch_is_skipped(self, patched_main):
        with patch("run.read_last_run", return_value="2024-01-01T00:00:00"), patch(
            "run.batch_generator", return_value=iter([([], "2024-01-01T01:00:00")])
        ), patch("run.process_batch") as mock_process:
            main([])
        mock_process.assert_not_called()

    def test_single_batch_calls_process_batch(self, patched_main, sample_payloads):
        with patch("run.read_last_run", return_value="2024-01-01T00:00:00"), patch(
            "run.batch_generator", return_value=iter([(sample_payloads, "2024-01-02T00:00:00")])
        ), patch("run.process_batch", return_value=(2, 0)) as mock_process:
            main([])
        mock_process.assert_called_once()
        _, _, _, payloads_arg, timestamp_arg = mock_process.call_args.args
        assert payloads_arg == sample_payloads
        assert timestamp_arg == "2024-01-02T00:00:00"

    def test_multiple_batches_accumulates_calls(self, patched_main, sample_payloads):
        batches = [
            (sample_payloads, "2024-01-02T00:00:00"),
            (sample_payloads, "2024-01-03T00:00:00"),
        ]
        with patch("run.read_last_run", return_value="2024-01-01T00:00:00"), patch(
            "run.batch_generator", return_value=iter(batches)
        ), patch("run.process_batch", return_value=(1, 0)) as mock_process:
            main([])
        assert mock_process.call_count == 2

    def test_last_run_used_in_query(self, patched_main):
        last_run = "2024-06-15T12:00:00"
        captured = {}

        def fake_batch_generator(es_client, query):
            captured["query"] = query
            return iter([])

        with patch("run.read_last_run", return_value=last_run), patch(
            "run.batch_generator", side_effect=fake_batch_generator
        ):
            main([])

        assert captured["query"]["query"]["range"]["processed"]["gt"] == last_run

    def test_batch_failure_aborts_run(self, patched_main, sample_payloads):
        batches = [
            (sample_payloads, "2024-01-02T00:00:00"),
            (sample_payloads, "2024-01-03T00:00:00"),
        ]
        with patch("run.read_last_run", return_value="2024-01-01T00:00:00"), patch(
            "run.batch_generator", return_value=iter(batches)
        ), patch("run.process_batch", return_value=(0, 2)) as mock_process:
            with pytest.raises(SystemExit) as exc_info:
                main([])
        # Fail-stop: continuing would advance the state file past the gap.
        assert mock_process.call_count == 1
        assert exc_info.value.code == 1

    def test_scan_failure_aborts_run(self, patched_main, sample_payloads):
        def batches_then_scan_error(es_client, query):
            yield (sample_payloads, "2024-01-02T00:00:00")
            raise RuntimeError("Elasticsearch scan failed mid-run")

        with patch("run.read_last_run", return_value="2024-01-01T00:00:00"), patch(
            "run.batch_generator", side_effect=batches_then_scan_error
        ), patch("run.process_batch", return_value=(2, 0)) as mock_process:
            with pytest.raises(SystemExit) as exc_info:
                main([])
        mock_process.assert_called_once()  # yielded batch was still processed
        assert exc_info.value.code == 1

    def test_progress_reported_when_scope_known(self, patched_main, sample_payloads):
        patched_main.es.return_value.count.return_value = {"count": 100}
        with patch("run.read_last_run", return_value="2024-01-01T00:00:00"), patch(
            "run.batch_generator", return_value=iter([(sample_payloads, "2024-01-02T00:00:00")])
        ), patch("run.process_batch", return_value=(2, 0)), patch("run.format_progress") as mock_fmt:
            main([])
        mock_fmt.assert_called_once()
        kwargs = mock_fmt.call_args.kwargs
        assert kwargs["total_docs"] == 2  # both sample_payloads source_ids
        assert kwargs["total_in_scope"] == 100
        assert kwargs["watermark"] == "2024-01-02T00:00:00"

    def test_clients_are_passed_to_process_batch(self, patched_main, sample_payloads):
        mock_qdrant = MagicMock()
        mock_cohere = MagicMock()
        mock_sparse = MagicMock()
        patched_main.qdrant.return_value = mock_qdrant
        patched_main.cohere.return_value = mock_cohere
        patched_main.sparse.return_value = mock_sparse
        with patch("run.read_last_run", return_value="2024-01-01T00:00:00"), patch(
            "run.batch_generator", return_value=iter([(sample_payloads, "2024-01-02T00:00:00")])
        ), patch("run.process_batch", return_value=(1, 0)) as mock_process:
            main([])
        qdrant_arg, cohere_arg, sparse_arg, _, _ = mock_process.call_args.args
        assert qdrant_arg is mock_qdrant
        assert cohere_arg is mock_cohere
        assert sparse_arg is mock_sparse


class TestSinceArgument:
    def test_since_produces_gte_query_and_skips_state_file(self, patched_main):
        captured = {}

        def fake_batch_generator(es_client, query):
            captured["query"] = query
            return iter([])

        with patch("run.read_last_run") as mock_read, patch("run.batch_generator", side_effect=fake_batch_generator):
            main(["--since", "2026-01-01T00:00:00"])

        assert captured["query"]["query"]["range"]["processed"]["gte"] == "2026-01-01T00:00:00"
        mock_read.assert_not_called()

    def test_invalid_since_exits(self):
        with pytest.raises(SystemExit):
            main(["--since", "not-a-date"])


class TestEstimateArgument:
    def test_estimate_dispatches_to_run_estimate(self, patched_main):
        with patch("run.run_estimate") as mock_estimate, patch("run.process_batch") as mock_process:
            main(["--estimate"])
        mock_estimate.assert_called_once()
        mock_process.assert_not_called()
        base_query = mock_estimate.call_args.args[4]
        assert base_query == {"range": {"processed": {"gte": EPOCH}}}

    def test_estimate_with_since(self, patched_main):
        with patch("run.run_estimate") as mock_estimate, patch("run.process_batch"):
            main(["--estimate", "--since", "2026-01-01T00:00:00"])
        base_query = mock_estimate.call_args.args[4]
        assert base_query == {"range": {"processed": {"gte": "2026-01-01T00:00:00"}}}

    def test_estimate_passes_sample_size_and_seed(self, patched_main):
        with patch("run.run_estimate") as mock_estimate, patch("run.process_batch"):
            main(["--estimate", "--sample-size", "123", "--seed", "7"])
        assert mock_estimate.call_args.args[5] == 123
        assert mock_estimate.call_args.args[6] == 7

    def test_no_dedup_skips_qdrant_client(self, patched_main):
        with patch("run.run_estimate") as mock_estimate, patch("run.process_batch"):
            main(["--estimate", "--no-dedup"])
        patched_main.qdrant.assert_not_called()
        assert mock_estimate.call_args.args[1] is None

    def test_estimate_without_cohere_key_still_runs(self, patched_main):
        patched_main.cohere.side_effect = ValueError("COHERE_API_KEY is not set.")
        with patch("run.run_estimate") as mock_estimate, patch("run.process_batch"):
            main(["--estimate"])
        mock_estimate.assert_called_once()
        assert mock_estimate.call_args.args[2] is None
