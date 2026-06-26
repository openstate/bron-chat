import logging
from typing import List, Generator, Tuple, Dict, Any

from cohere import ClientV2
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from app.partitioning import html_partition, html_txt_partition, txt_partition
from app.utils import remove_processing_instructions
from app.embedding import generate_dense_embeddings, generate_sparse_embedding
from app.qdrant_handler import filter_new_chunks, make_qdrant_points, upsert_points_to_qdrant
from app.state import write_last_run
from app.config import settings

logger = logging.getLogger(__name__)


def prepare_qdrant_payload(doc: dict) -> Tuple[List[dict] | None, str | None]:
    """Split an ES document into chunk payloads. Returns (payloads, processed_timestamp)."""
    qdrant_payload = []

    try:
        description = doc.get("_source", {}).get("description", "")
        processed = doc.get("_source", {}).get("processed", "")

        try:
            description = remove_processing_instructions(description)
        except Exception:
            pass

        source = doc.get("_source", {}).get("source", "")
        title = doc["_source"].get("title", "")
        location_name = doc["_source"].get("location_name", "")
        doc_url = doc["_source"].get("doc_url", "")
        location = doc["_source"].get("location", "")
        modified = doc["_source"].get("modified", "")
        published = doc["_source"].get("published", "")
        doc_type = doc["_source"].get("type", "")
        identifier = doc["_source"].get("identifier", "")
        url = doc["_source"].get("url", "")
        source_id = doc["_id"]

        markdown_chunks = []
        try:
            if source in ("cvdr", "poliflw"):
                markdown_chunks = html_partition(description)
            elif source in ("oor", "woogle", "obk"):
                markdown_chunks = txt_partition(description)
            elif source == "openbesluitvorming":
                markdown_chunks = html_txt_partition(description)
            else:
                markdown_chunks = txt_partition(description)
        except Exception as e:
            logger.error(f"Partitioning failed for doc ID {source_id}: {e}")
            return None, processed

        chunk_count = len(markdown_chunks)

        for i, markdown_chunk in enumerate(markdown_chunks):
            qdrant_payload.append(
                {
                    "content": markdown_chunk,
                    "meta": {
                        "title": title,
                        "location": location,
                        "location_name": location_name,
                        "modified": modified,
                        "published": published,
                        "type": doc_type,
                        "identifier": identifier,
                        "url": url,
                        "doc_url": doc_url,
                        "source": source,
                        "source_id": source_id,
                        "page_number": i,
                        "page_count": chunk_count,
                    },
                }
            )

        return qdrant_payload, processed

    except Exception as e:
        logger.error(f"Unexpected error preparing payload for doc: {e}")
        return None, None


def batch_generator(
    es_client: Elasticsearch,
    query: dict,
    max_documents: int | None = None,
) -> Generator[Tuple[List[Dict[str, Any]], str], None, None]:
    """Scan ES and yield (payloads, max_processed_in_batch).

    The scan is sorted by `processed` ascending so the batch max is a true
    watermark to resume from. If the scan fails, the partial batch is still
    yielded before the error is re-raised."""
    if max_documents is None:
        max_documents = settings.MAX_DOCUMENTS_TO_PROCESS

    processing_threshold = settings.BATCH_SIZE * settings.PROCESSING_THRESHOLD_MULTIPLIER

    query = {**query, "sort": [{"processed": {"order": "asc"}}]}

    batch_payloads: List[Dict[str, Any]] = []
    batch_max_processed = ""
    documents_processed_count = 0
    scan_error: Exception | None = None

    logger.info("Starting Elasticsearch scan...")
    try:
        for doc in scan(
            client=es_client,
            index=settings.ES_INDEX,
            query=query,
            scroll=settings.ES_SCROLL_TIME,
            size=settings.BATCH_SIZE,
            preserve_order=True,
        ):
            if max_documents != -1 and documents_processed_count >= max_documents:
                logger.info(f"Maximum document limit ({max_documents}) reached.")
                break

            payloads, doc_processed = prepare_qdrant_payload(doc)

            if payloads:
                batch_payloads.extend(payloads)
                documents_processed_count += 1

                if doc_processed and doc_processed > batch_max_processed:
                    batch_max_processed = doc_processed

            if len(batch_payloads) >= processing_threshold:
                yield batch_payloads, batch_max_processed
                batch_payloads = []
                batch_max_processed = ""

    except Exception as e:
        scan_error = e
        logger.critical(f"Critical error during Elasticsearch scan: {e}")

    finally:
        if batch_payloads:
            yield batch_payloads, batch_max_processed
        logger.info("Elasticsearch scan finished.")

    if scan_error is not None:
        raise RuntimeError("Elasticsearch scan failed mid-run") from scan_error


def process_batch(
    qdrant_client: QdrantClient,
    cohere_client: ClientV2,
    sparse_embedder: SparseTextEmbedding,
    payloads: List[Dict[str, Any]],
    batch_max_processed: str,
) -> Tuple[int, int]:
    """Embed chunks not yet in Qdrant, upsert them, and advance the state file.

    Returns (upserted_points, errors)."""
    if not payloads:
        return 0, 0

    logger.info(f"Processing batch of {len(payloads)} chunks...")

    new_pairs = filter_new_chunks(qdrant_client, payloads)

    if not new_pairs:
        logger.info("All chunks in batch already exist in Qdrant, skipping.")
        if batch_max_processed:
            write_last_run(batch_max_processed)
        return 0, 0

    texts_to_embed = [p["content"] for _, p in new_pairs]

    sparse_vectors = generate_sparse_embedding(sparse_embedder, texts_to_embed)
    if sparse_vectors is None:
        logger.error("Failed to generate sparse embeddings. Skipping batch.")
        return 0, len(new_pairs)

    dense_vectors = generate_dense_embeddings(cohere_client, texts_to_embed)
    if dense_vectors is None:
        logger.error("Failed to generate dense embeddings. Skipping batch.")
        return 0, len(new_pairs)

    qdrant_points = make_qdrant_points(new_pairs, dense_vectors, sparse_vectors)

    if not qdrant_points:
        logger.warning("No valid Qdrant points created from batch.")
        return 0, len(new_pairs)

    try:
        upsert_points_to_qdrant(qdrant_client, qdrant_points)
        if batch_max_processed:
            write_last_run(batch_max_processed)
        return len(qdrant_points), 0
    except Exception as e:
        logger.error(f"Qdrant upsert failed: {e}")
        return 0, len(qdrant_points)
