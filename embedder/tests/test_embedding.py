from unittest.mock import MagicMock, call, patch

from app.embedding import _sparse_embed_texts, generate_dense_embeddings, generate_sparse_embedding

PATCH = "app.embedding.{}"


def _mock_settings(**kwargs):
    defaults = {
        "DENSE_BATCH_SIZE": 96,
        "COHERE_RETRIES": 3,
        "COHERE_RETRY_DELAY": 1,
        "COHERE_INPUT_TYPE": "search_document",
        "COHERE_EMBED_MODEL": "embed-multilingual-v3.0",
        "SPARSE_BATCH_SIZE": 32,
    }
    defaults.update(kwargs)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _cohere_response(vectors):
    response = MagicMock()
    response.embeddings.uint8 = vectors
    return response


class TestGenerateDenseEmbeddings:
    def test_success_returns_embeddings(self):
        client = MagicMock()
        client.embed.return_value = _cohere_response([[1, 2], [3, 4]])
        with patch(PATCH.format("settings"), _mock_settings()):
            result = generate_dense_embeddings(client, ["a", "b"])
        assert result == [[1, 2], [3, 4]]

    def test_batches_texts_by_dense_batch_size(self):
        client = MagicMock()
        client.embed.side_effect = [
            _cohere_response([[1, 2]]),
            _cohere_response([[3, 4]]),
        ]
        with patch(PATCH.format("settings"), _mock_settings(DENSE_BATCH_SIZE=1)):
            result = generate_dense_embeddings(client, ["a", "b"])
        assert client.embed.call_count == 2
        assert result == [[1, 2], [3, 4]]

    def test_retries_on_transient_error_then_succeeds(self):
        client = MagicMock()
        client.embed.side_effect = [Exception("timeout"), _cohere_response([[1, 2]])]
        with patch(PATCH.format("settings"), _mock_settings(COHERE_RETRIES=3)), \
             patch(PATCH.format("time")):
            result = generate_dense_embeddings(client, ["a"])
        assert result == [[1, 2]]
        assert client.embed.call_count == 2

    def test_exhausted_retries_returns_none(self):
        client = MagicMock()
        client.embed.side_effect = Exception("always fails")
        with patch(PATCH.format("settings"), _mock_settings(COHERE_RETRIES=2)), \
             patch(PATCH.format("time")):
            result = generate_dense_embeddings(client, ["a"])
        assert result is None
        assert client.embed.call_count == 2

    def test_none_uint8_response_returns_none(self):
        client = MagicMock()
        client.embed.return_value = _cohere_response(None)
        with patch(PATCH.format("settings"), _mock_settings(COHERE_RETRIES=1)), \
             patch(PATCH.format("time")):
            result = generate_dense_embeddings(client, ["a"])
        assert result is None

    def test_empty_texts_returns_empty_list(self):
        client = MagicMock()
        with patch(PATCH.format("settings"), _mock_settings()):
            result = generate_dense_embeddings(client, [])
        assert result == []
        client.embed.assert_not_called()

    def test_backoff_grows_exponentially(self):
        client = MagicMock()
        client.embed.side_effect = [Exception("fail"), Exception("fail"), _cohere_response([[1]])]
        with patch(PATCH.format("settings"), _mock_settings(COHERE_RETRIES=3, COHERE_RETRY_DELAY=5)), \
             patch(PATCH.format("time")) as mock_time:
            generate_dense_embeddings(client, ["a"])
        mock_time.sleep.assert_has_calls([call(5), call(10)])


class TestSparseEmbedTexts:
    def test_returns_all_embeddings(self):
        embedder = MagicMock()
        emb1, emb2 = MagicMock(), MagicMock()
        embedder.embed.return_value = [emb1, emb2]
        progress_bar = MagicMock()
        result = _sparse_embed_texts(embedder, ["a", "b"], progress_bar, batch_size=10)
        assert result == [emb1, emb2]

    def test_batches_by_batch_size(self):
        embedder = MagicMock()
        embedder.embed.side_effect = [[MagicMock()], [MagicMock()]]
        progress_bar = MagicMock()
        _sparse_embed_texts(embedder, ["a", "b"], progress_bar, batch_size=1)
        assert embedder.embed.call_count == 2

    def test_error_in_batch_continues(self):
        embedder = MagicMock()
        good_emb = MagicMock()
        embedder.embed.side_effect = [Exception("fail"), [good_emb]]
        progress_bar = MagicMock()
        result = _sparse_embed_texts(embedder, ["a", "b"], progress_bar, batch_size=1)
        assert result == [good_emb]


class TestGenerateSparseEmbedding:
    def test_empty_texts_returns_empty_list(self):
        result = generate_sparse_embedding(MagicMock(), [])
        assert result == []

    def test_success_returns_embeddings(self):
        sparse_embedder = MagicMock()
        emb = MagicMock()
        sparse_embedder.embed.return_value = [emb]
        with patch(PATCH.format("settings"), _mock_settings()):
            result = generate_sparse_embedding(sparse_embedder, ["text"])
        assert result is not None
        assert emb in result

    def test_unexpected_error_returns_none(self):
        with patch(PATCH.format("settings"), _mock_settings()), \
             patch(PATCH.format("_sparse_embed_texts"), side_effect=Exception("boom")):
            result = generate_sparse_embedding(MagicMock(), ["text"])
        assert result is None
