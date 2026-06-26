import json
import logging
import math
import os
import random
import statistics
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from cohere import ClientV2
from elasticsearch import Elasticsearch
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from app.config import settings
from app.embedding import generate_sparse_embedding
from app.processing import prepare_qdrant_payload
from app.qdrant_handler import filter_new_chunks

logger = logging.getLogger(__name__)

MISSING_SOURCE = "__missing__"

# Qdrant stores sparse vectors as (u32 index, f32 value) pairs.
SPARSE_BYTES_PER_ELEMENT = 8

# embed-multilingual-light-v3.0: 384 dims stored as uint8.
FALLBACK_DENSE_BYTES = 384 * 1

DATATYPE_BYTES = {"uint8": 1, "float16": 2, "float32": 4}

DEDUP_SLICE = 2000


@dataclass
class SourceEstimate:
    """Sample statistics for one source stratum.

    per_doc_tokens / per_doc_bytes hold one entry per sampled document;
    fully-deduped and failed docs stay as zeros to keep the mean unbiased."""

    source: str
    n_total: int
    n_sampled: int
    chunk_count: int
    new_chunk_count: int
    per_doc_tokens: List[int]
    per_doc_bytes: List[int]
    partition_failures: int

    @property
    def est_tokens(self) -> float:
        return extrapolate(self.per_doc_tokens, self.n_total)

    @property
    def est_bytes(self) -> float:
        return extrapolate(self.per_doc_bytes, self.n_total)

    @property
    def est_points(self) -> float:
        if not self.n_sampled:
            return 0.0
        return self.new_chunk_count / self.n_sampled * self.n_total


def get_source_counts(es_client: Elasticsearch, base_query: dict) -> Dict[str, int]:
    """Count in-scope documents per source via a terms aggregation."""
    response = es_client.search(
        index=settings.ES_INDEX,
        size=0,
        query=base_query,
        aggs={
            "by_source": {
                "terms": {
                    "field": settings.ESTIMATE_SOURCE_FIELD,
                    "size": 100,
                    "missing": MISSING_SOURCE,
                }
            }
        },
    )
    buckets = response["aggregations"]["by_source"]["buckets"]
    return {b["key"]: b["doc_count"] for b in buckets if b["doc_count"] > 0}


def sample_docs(
    es_client: Elasticsearch,
    source: str,
    base_query: dict,
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Draw a seeded uniform random sample of documents for one source."""
    if source == MISSING_SOURCE:
        source_filter: dict = {
            "bool": {"must_not": {"exists": {"field": settings.ESTIMATE_SOURCE_FIELD}}}
        }
    else:
        source_filter = {"term": {settings.ESTIMATE_SOURCE_FIELD: source}}

    response = es_client.search(
        index=settings.ES_INDEX,
        size=n,
        query={
            "function_score": {
                "query": {"bool": {"filter": [base_query, source_filter]}},
                "random_score": {"seed": seed, "field": "_seq_no"},
                "boost_mode": "replace",
            }
        },
    )
    return response["hits"]["hits"]


def load_tokenizer(cohere_client: ClientV2 | None):
    """Load the Cohere tokenizer for offline token counting, cached in MODELS_DIR.

    Returns None on failure; callers fall back to a chars-per-token heuristic."""
    cache_path = os.path.join(
        settings.MODELS_DIR, f"cohere-{settings.COHERE_EMBED_MODEL}-tokenizer.json"
    )
    try:
        from tokenizers import Tokenizer

        if os.path.exists(cache_path):
            return Tokenizer.from_file(cache_path)

        if cohere_client is None:
            return None

        model_info = cohere_client.models.get(settings.COHERE_EMBED_MODEL)
        tokenizer_url = getattr(model_info, "tokenizer_url", None)
        if not isinstance(tokenizer_url, str) or not tokenizer_url:
            return None

        with urllib.request.urlopen(tokenizer_url) as response:
            tokenizer_json = response.read().decode("utf-8")

        try:
            os.makedirs(settings.MODELS_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                f.write(tokenizer_json)
        except Exception as e:
            logger.warning(f"Could not cache tokenizer at {cache_path}: {e}")

        return Tokenizer.from_str(tokenizer_json)
    except Exception as e:
        logger.warning(
            f"Could not load Cohere tokenizer ({e}); "
            f"falling back to chars/{settings.ESTIMATE_CHARS_PER_TOKEN} heuristic."
        )
        return None


def count_billable_tokens(tokenizer, text: str) -> int:
    """Token count for one chunk, capped at COHERE_MAX_TOKENS (Cohere truncates the rest)."""
    if tokenizer is not None:
        token_count = len(tokenizer.encode(text).ids)
    else:
        token_count = math.ceil(len(text) / settings.ESTIMATE_CHARS_PER_TOKEN)
    return min(token_count, settings.COHERE_MAX_TOKENS)


def get_dense_vector_bytes(qdrant_client: QdrantClient | None) -> Tuple[int, str]:
    """Bytes per stored dense vector, from the collection config (Qdrant stores
    per the declared datatype, not what the client sends).
    Returns (bytes_per_vector, description)."""
    if qdrant_client is None:
        return FALLBACK_DENSE_BYTES, "384 x uint8 (assumed, no Qdrant connection)"
    try:
        info = qdrant_client.get_collection(settings.QDRANT_COLLECTION)
        vectors = info.config.params.vectors
        dense = vectors.get("text-dense") if isinstance(vectors, dict) else vectors
        if dense is None:
            raise ValueError("collection has no 'text-dense' vector configured")
        dims = dense.size
        datatype = getattr(dense, "datatype", None)
        datatype_str = str(getattr(datatype, "value", datatype) or "float32").lower()
        bytes_per_dim = DATATYPE_BYTES.get(datatype_str, 4)
        return dims * bytes_per_dim, f"{dims} x {datatype_str} (from collection)"
    except Exception as e:
        logger.warning(f"Could not read collection config ({e}); assuming 384 x uint8.")
        return FALLBACK_DENSE_BYTES, "384 x uint8 (assumed)"


def point_bytes(payload: Dict[str, Any], sparse_nonzeros: int, dense_bytes: int) -> int:
    """Raw bytes of one point (dense + sparse + payload), excluding index overhead."""
    payload_size = len(json.dumps(payload).encode("utf-8"))
    return dense_bytes + sparse_nonzeros * SPARSE_BYTES_PER_ELEMENT + payload_size


def _sparse_nonzero_counts(
    sparse_embedder: SparseTextEmbedding | None, payloads: List[Dict[str, Any]]
) -> List[int]:
    """Nonzero elements per chunk; falls back to unique-word counts when sparse
    embedding fails or returns fewer items than texts (which would misalign the zip)."""
    if not payloads:
        return []
    if sparse_embedder is not None:
        sparse_vectors = generate_sparse_embedding(
            sparse_embedder, [p["content"] for p in payloads]
        )
        if sparse_vectors and len(sparse_vectors) == len(payloads):
            return [len(v.indices) for v in sparse_vectors]
        logger.warning("Sparse embedding incomplete; approximating nonzeros by unique words.")
    return [len(set(p["content"].split())) for p in payloads]


def estimate_source(
    es_client: Elasticsearch,
    qdrant_client: QdrantClient | None,
    sparse_embedder: SparseTextEmbedding | None,
    tokenizer,
    source: str,
    n_total: int,
    base_query: dict,
    sample_size: int,
    seed: int,
    dense_bytes: int,
) -> SourceEstimate:
    """Sample one source and run it through the real pipeline, read-only."""
    docs = sample_docs(es_client, source, base_query, min(sample_size, n_total), seed)

    partition_failures = 0
    sampled_ids: List[str] = []
    all_payloads: List[Dict[str, Any]] = []

    for doc in docs:
        sampled_ids.append(doc["_id"])
        payloads, _ = prepare_qdrant_payload(doc)
        if payloads is None:
            partition_failures += 1
            continue
        all_payloads.extend(payloads)

    if qdrant_client is not None:
        new_payloads: List[Dict[str, Any]] = []
        for i in range(0, len(all_payloads), DEDUP_SLICE):
            pairs = filter_new_chunks(qdrant_client, all_payloads[i : i + DEDUP_SLICE])
            new_payloads.extend(p for _, p in pairs)
    else:
        new_payloads = all_payloads

    nonzero_counts = _sparse_nonzero_counts(sparse_embedder, new_payloads)

    per_doc_tokens = {source_id: 0 for source_id in sampled_ids}
    per_doc_bytes = {source_id: 0 for source_id in sampled_ids}

    for payload, nonzeros in zip(new_payloads, nonzero_counts):
        source_id = payload["meta"]["source_id"]
        per_doc_tokens[source_id] += count_billable_tokens(tokenizer, payload["content"])
        per_doc_bytes[source_id] += point_bytes(payload, nonzeros, dense_bytes)

    return SourceEstimate(
        source=source,
        n_total=n_total,
        n_sampled=len(docs),
        chunk_count=len(all_payloads),
        new_chunk_count=len(new_payloads),
        per_doc_tokens=list(per_doc_tokens.values()),
        per_doc_bytes=list(per_doc_bytes.values()),
        partition_failures=partition_failures,
    )


def extrapolate(per_doc_values: List[int], n_total: int) -> float:
    """Point estimate of the population total: sample mean x population size."""
    if not per_doc_values:
        return 0.0
    return statistics.fmean(per_doc_values) * n_total


def bootstrap_ci(
    strata: List[Tuple[List[int], int]],
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for the summed extrapolated total over
    (per_doc_values, n_total) strata."""
    strata = [(values, n_total) for values, n_total in strata if values]
    if not strata:
        return 0.0, 0.0

    rng = random.Random(seed)
    totals = []
    for _ in range(iterations):
        total = 0.0
        for values, n_total in strata:
            resample = rng.choices(values, k=len(values))
            total += statistics.fmean(resample) * n_total
        totals.append(total)

    totals.sort()
    low = totals[int(0.025 * (len(totals) - 1))]
    high = totals[int(0.975 * (len(totals) - 1))]
    return low, high


def _fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def _fmt_tokens(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def _fmt_money(tokens: float) -> str:
    return f"${tokens / 1e6 * settings.COHERE_PRICE_PER_1M_TOKENS:,.2f}"


def _fmt_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def run_estimate(
    es_client: Elasticsearch,
    qdrant_client: QdrantClient | None,
    cohere_client: ClientV2 | None,
    sparse_embedder: SparseTextEmbedding | None,
    base_query: dict,
    sample_size: int,
    seed: int,
) -> List[SourceEstimate]:
    """Estimate Cohere cost and Qdrant storage growth for the in-scope documents.

    Read-only: nothing is embedded, upserted, or written to the state file."""
    tokenizer = load_tokenizer(cohere_client)
    dense_bytes, dense_desc = get_dense_vector_bytes(qdrant_client)

    source_counts = get_source_counts(es_client, base_query)
    if not source_counts:
        logger.info("No documents in scope; nothing to estimate.")
        return []

    estimates: List[SourceEstimate] = []
    for source, n_total in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        logger.info(f"Sampling source '{source}' ({n_total} docs in scope)...")
        estimates.append(
            estimate_source(
                es_client,
                qdrant_client,
                sparse_embedder,
                tokenizer,
                source,
                n_total,
                base_query,
                sample_size,
                seed,
                dense_bytes,
            )
        )

    iterations = settings.ESTIMATE_BOOTSTRAP_ITERATIONS

    header = (
        f"{'source':<20} {'docs':>10} {'sampled':>8} {'chunks/doc':>10} "
        f"{'in-qdrant':>9} {'est. tokens':>26} {'est. cost':>22} "
        f"{'est. points':>11} {'est. storage':>12}"
    )
    lines = ["", "--- Cost estimate (sampled) ---", header, "-" * len(header)]

    for est in estimates:
        chunks_per_doc = est.chunk_count / est.n_sampled if est.n_sampled else 0.0
        deduped_pct = (
            (est.chunk_count - est.new_chunk_count) / est.chunk_count * 100
            if est.chunk_count
            else 0.0
        )
        token_low, token_high = bootstrap_ci(
            [(est.per_doc_tokens, est.n_total)], iterations, seed
        )
        lines.append(
            f"{est.source:<20} {_fmt_count(est.n_total):>10} {est.n_sampled:>8} "
            f"{chunks_per_doc:>10.1f} {deduped_pct:>8.0f}% "
            f"{_fmt_tokens(est.est_tokens) + ' [' + _fmt_tokens(token_low) + '-' + _fmt_tokens(token_high) + ']':>26} "
            f"{_fmt_money(est.est_tokens) + ' [' + _fmt_money(token_low) + '-' + _fmt_money(token_high) + ']':>22} "
            f"{_fmt_count(est.est_points):>11} {_fmt_bytes(est.est_bytes):>12}"
        )

    total_tokens = sum(est.est_tokens for est in estimates)
    total_points = sum(est.est_points for est in estimates)
    total_bytes = sum(est.est_bytes for est in estimates)
    token_strata = [(est.per_doc_tokens, est.n_total) for est in estimates]
    byte_strata = [(est.per_doc_bytes, est.n_total) for est in estimates]
    token_low, token_high = bootstrap_ci(token_strata, iterations, seed)
    bytes_low, bytes_high = bootstrap_ci(byte_strata, iterations, seed)

    total_docs = sum(est.n_total for est in estimates)
    total_sampled = sum(est.n_sampled for est in estimates)
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<20} {_fmt_count(total_docs):>10} {total_sampled:>8} "
        f"{'':>10} {'':>9} "
        f"{_fmt_tokens(total_tokens) + ' [' + _fmt_tokens(token_low) + '-' + _fmt_tokens(token_high) + ']':>26} "
        f"{_fmt_money(total_tokens) + ' [' + _fmt_money(token_low) + '-' + _fmt_money(token_high) + ']':>22} "
        f"{_fmt_count(total_points):>11} "
        f"{_fmt_bytes(total_bytes) + ' [' + _fmt_bytes(bytes_low) + '-' + _fmt_bytes(bytes_high) + ']':>12}"
    )

    tokenizer_desc = "exact (cohere)" if tokenizer is not None else (
        f"HEURISTIC chars/{settings.ESTIMATE_CHARS_PER_TOKEN}"
    )
    lines.append(
        f"rate: ${settings.COHERE_PRICE_PER_1M_TOKENS}/1M tokens | "
        f"tokenizer: {tokenizer_desc} | dense: {dense_desc} | seed {seed}"
    )

    total_failures = sum(est.partition_failures for est in estimates)
    if total_failures:
        lines.append(f"warnings: {total_failures} partition failures in sample")

    logger.info("\n".join(lines))
    return estimates
