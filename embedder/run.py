import argparse
import logging
import time
from datetime import datetime, timedelta

from app.clients import (
    get_cohere_client,
    get_elasticsearch_client,
    get_qdrant_client,
    get_sparse_embedder,
)
from app.config import settings
from app.estimate import run_estimate
from app.processing import batch_generator, process_batch
from app.state import EPOCH, read_last_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def iso_datetime(value: str) -> str:
    """Argparse type: validate an ISO-8601 datetime, pass the original string to ES."""
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an ISO-8601 datetime (e.g. 2026-01-01T00:00:00), got {value!r}")
    return value


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed documents from Elasticsearch into Qdrant.")
    parser.add_argument(
        "--since",
        type=iso_datetime,
        default=None,
        help="Only consider documents with processed >= this ISO-8601 datetime. "
        "Default: the state file (normal mode) or the full corpus (estimate mode).",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate Cohere cost and Qdrant storage growth from a random sample "
        "instead of embedding. Read-only: nothing is embedded, upserted, or "
        "written to the state file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.ESTIMATE_SAMPLE_SIZE,
        help="Estimate mode: documents sampled per source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.ESTIMATE_SEED,
        help="Estimate mode: random seed for sampling and bootstrap.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Estimate mode: skip the Qdrant existence check (worst-case / fresh-collection cost).",
    )
    return parser.parse_args(argv)


def run_estimate_mode(es_client, args: argparse.Namespace) -> None:
    qdrant_client = None if args.no_dedup else get_qdrant_client()

    cohere_client = None
    try:
        cohere_client = get_cohere_client()
    except ValueError as e:
        logger.warning(f"{e} Token counts will use the chars-per-token heuristic.")

    sparse_embedder = get_sparse_embedder()

    since = args.since or EPOCH
    logger.info(f"Estimating documents with processed >= {since}")
    base_query = {"range": {"processed": {"gte": since}}}

    run_estimate(
        es_client,
        qdrant_client,
        cohere_client,
        sparse_embedder,
        base_query,
        args.sample_size,
        args.seed,
    )


def format_progress(total_docs, total_in_scope, watermark, elapsed_seconds, remaining_seconds):
    """Render the per-batch progress line. Pure so the percentage/ETA math is testable."""
    pct = total_docs / total_in_scope * 100
    return (
        f"Progress: {total_docs:,}/{total_in_scope:,} docs ({pct:.1f}%) | "
        f"watermark {watermark} | "
        f"elapsed {timedelta(seconds=int(elapsed_seconds))} | "
        f"ETA {timedelta(seconds=int(remaining_seconds))}"
    )


def main(argv=None):
    args = parse_args(argv)

    es_client = get_elasticsearch_client()

    if args.estimate:
        run_estimate_mode(es_client, args)
        return

    qdrant_client = get_qdrant_client()
    cohere_client = get_cohere_client()
    sparse_embedder = get_sparse_embedder()

    if args.since:
        logger.info(f"Processing documents with processed >= {args.since} (--since)")
        query = {"query": {"range": {"processed": {"gte": args.since}}}}
    else:
        last_run = read_last_run()
        logger.info(f"Processing documents with processed > {last_run}")
        query = {"query": {"range": {"processed": {"gt": last_run}}}}

    try:
        total_in_scope = int(es_client.count(index=settings.ES_INDEX, query=query["query"])["count"])
        logger.info(f"{total_in_scope:,} documents in scope.")
    except Exception as e:
        total_in_scope = 0
        logger.warning(f"Could not count documents in scope (no ETA reporting): {e}")

    start_time = time.monotonic()
    total_docs = 0
    total_chunks = 0
    total_upserted = 0
    total_errors = 0
    aborted = False

    try:
        for batch_payloads, batch_max_processed in batch_generator(es_client, query):
            if not batch_payloads:
                continue

            unique_source_ids = len({p["meta"]["source_id"] for p in batch_payloads})
            total_docs += unique_source_ids
            total_chunks += len(batch_payloads)

            upserted, errors = process_batch(
                qdrant_client,
                cohere_client,
                sparse_embedder,
                batch_payloads,
                batch_max_processed,
            )
            total_upserted += upserted
            total_errors += errors

            if errors:
                logger.critical("Batch failed after retries; aborting run. Rerun to resume from the state file.")
                aborted = True
                break

            if total_in_scope and total_docs:
                elapsed = time.monotonic() - start_time
                remaining = elapsed / total_docs * max(total_in_scope - total_docs, 0)
                logger.info(
                    format_progress(
                        total_docs=total_docs,
                        total_in_scope=total_in_scope,
                        watermark=batch_max_processed,
                        elapsed_seconds=elapsed,
                        remaining_seconds=remaining,
                    )
                )
    except Exception as e:
        logger.critical(f"Run aborted: {e}")
        aborted = True

    logger.info("\n--- Embedding run summary ---")
    logger.info(f"Documents processed:  {total_docs}")
    logger.info(f"Chunks prepared:      {total_chunks}")
    logger.info(f"Points upserted:      {total_upserted}")
    logger.info(f"Errors:               {total_errors}")

    if aborted:
        logger.critical("Run did NOT complete; rerun to resume from the state file.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
