import time
import logging
from typing import List
from cohere import ClientV2
from fastembed import SparseTextEmbedding
from tqdm import tqdm
from fastembed.sparse import SparseEmbedding

from app.config import settings

logger = logging.getLogger(__name__)


def generate_dense_embeddings(
    cohere_client: ClientV2, texts: List[str]
) -> List[List[int]] | None:
    embeddings: List[List[int]] = []

    for i in range(0, len(texts), settings.DENSE_BATCH_SIZE):
        batch_texts = texts[i : i + settings.DENSE_BATCH_SIZE]

        for attempt in range(settings.COHERE_RETRIES):
            try:
                response = cohere_client.embed(
                    texts=batch_texts,
                    input_type=settings.COHERE_INPUT_TYPE,
                    model=settings.COHERE_EMBED_MODEL,
                    embedding_types=["uint8"],
                )

                if response.embeddings.uint8 is None:
                    raise ValueError("Cohere returned None for uint8 embeddings.")

                embeddings.extend(response.embeddings.uint8)
                break
            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1} failed with error: {type(e).__name__}: {str(e)}"
                )
                if attempt < settings.COHERE_RETRIES - 1:
                    time.sleep(settings.COHERE_RETRY_DELAY * (2**attempt))
                else:
                    logger.error(
                        f"Failed to generate dense embeddings for batch at index {i} after {settings.COHERE_RETRIES} attempts."
                    )
                    return None

    return embeddings


def _sparse_embed_texts(
    embedder, texts: List[str], progress_bar: tqdm, batch_size: int
) -> List[SparseEmbedding]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_docs = texts[i : i + batch_size]
        try:
            for embedding in embedder.embed(batch_docs):
                embeddings.append(embedding)
                progress_bar.update(1)
        except Exception as e:
            logger.error(f"Error embedding sparse batch at index {i}: {e}")
    return embeddings


def generate_sparse_embedding(
    sparse_embedder: SparseTextEmbedding, texts: List[str]
) -> List[SparseEmbedding] | None:
    if not texts:
        return []

    try:
        with tqdm(
            total=len(texts),
            desc="Generating sparse embeddings",
            position=1,
            leave=False,
        ) as progress_bar:
            return _sparse_embed_texts(
                sparse_embedder, texts, progress_bar, settings.SPARSE_BATCH_SIZE
            )
    except Exception as e:
        logger.error(f"Failed to generate sparse embeddings: {e}")
        return None
