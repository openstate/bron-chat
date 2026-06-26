import uuid
import logging
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from fastembed.sparse import SparseEmbedding
from tqdm import tqdm

from app.config import settings

logger = logging.getLogger(__name__)


def _chunk_id(source_id: str, chunk_index: int) -> str:
    """Deterministic UUID for a chunk, so upserts are idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}#{chunk_index}"))


def filter_new_chunks(
    qdrant_client: QdrantClient,
    payloads: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return only (chunk_id, payload) pairs whose point does not yet exist in Qdrant."""
    id_payload_pairs = [
        (_chunk_id(p["meta"]["source_id"], p["meta"]["page_number"]), p)
        for p in payloads
    ]

    all_ids = [pair[0] for pair in id_payload_pairs]

    existing = qdrant_client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=all_ids,
        with_payload=False,
        with_vectors=False,
    )
    existing_ids = {str(point.id) for point in existing}

    new_pairs = [(cid, p) for cid, p in id_payload_pairs if cid not in existing_ids]
    skipped = len(id_payload_pairs) - len(new_pairs)
    if skipped:
        logger.info(f"Skipped {skipped} chunks already in Qdrant.")

    return new_pairs


def make_qdrant_points(
    id_payload_pairs: List[Tuple[str, Dict[str, Any]]],
    dense_vectors: List[List[int]],
    sparse_vectors: List[SparseEmbedding],
) -> List[PointStruct]:
    points = []

    if not (len(id_payload_pairs) == len(dense_vectors) == len(sparse_vectors)):
        logger.error("Mismatched lengths of id_payload_pairs, dense vectors, and sparse vectors.")
        return []

    for (point_id, payload), dense_vector, sparse_vector in zip(
        id_payload_pairs, dense_vectors, sparse_vectors
    ):
        try:
            if not isinstance(sparse_vector, SparseEmbedding):
                logger.error(f"Unexpected sparse vector format: {type(sparse_vector)}. Aborting batch.")
                return []

            sparse_dict = SparseVector(
                indices=[x for x in sparse_vector.indices],
                values=[float(x) for x in sparse_vector.values],
            )

            point = PointStruct(
                id=point_id,
                payload=payload,
                vector={
                    "text-dense": [float(x) for x in dense_vector],
                    "text-sparse": sparse_dict,
                },
            )
            points.append(point)
        except Exception as e:
            logger.error(f"Error creating Qdrant point for source_id {payload.get('meta', {}).get('source_id')}: {e}")
            continue

    return points


def upsert_points_to_qdrant(
    qdrant_client: QdrantClient,
    points: List[PointStruct],
    collection_name: str | None = None,
    batch_size: int | None = None,
) -> None:
    """Upsert Qdrant points in batches. Raises on upsert failure."""
    collection_name = collection_name or settings.QDRANT_COLLECTION
    batch_size = batch_size or settings.QDRANT_UPSERT_BATCH_SIZE

    if not points:
        logger.info("No points to upsert.")
        return

    total_points = len(points)
    logger.info(f"Upserting {total_points} points to '{collection_name}'...")

    try:
        with tqdm(total=total_points, desc="Upserting points", position=1, leave=False) as progress_bar:
            for i in range(0, total_points, batch_size):
                batch = points[i : i + batch_size]
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=batch,
                    )
                    progress_bar.update(len(batch))
                except Exception as e:
                    logger.error(f"Error upserting batch at index {i}: {e}")
                    raise
        logger.info("Upsert complete.")
    except Exception as e:
        logger.error(f"Upsert process failed: {e}")
        raise
