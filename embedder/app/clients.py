import logging
import cohere
import onnxruntime as ort
from qdrant_client import QdrantClient
from fastembed.sparse import SparseTextEmbedding
from elasticsearch import Elasticsearch

from app.config import settings

logger = logging.getLogger(__name__)


def get_elasticsearch_client() -> Elasticsearch:
    es_client = Elasticsearch(
        settings.ES_HOST,
        max_retries=settings.ES_MAX_RETRIES,
        retry_on_timeout=True,
        request_timeout=settings.ES_TIMEOUT,
    )
    logger.info(es_client.info())
    if not es_client.indices.exists(index=settings.ES_INDEX):
        raise ValueError(f"Elasticsearch index '{settings.ES_INDEX}' does not exist.")
    return es_client


def get_qdrant_client() -> QdrantClient:
    qdrant_client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=settings.QDRANT_TIMEOUT,
    )
    try:
        qdrant_client.get_collection(settings.QDRANT_COLLECTION)
    except Exception:
        raise ValueError(
            f"Qdrant collection '{settings.QDRANT_COLLECTION}' does not exist."
        )
    return qdrant_client


def get_cohere_client() -> cohere.ClientV2:
    if not settings.COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY is not set.")
    return cohere.ClientV2(api_key=settings.COHERE_API_KEY)


def get_sparse_embedder() -> SparseTextEmbedding:
    session_options = ort.SessionOptions()
    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = settings.NUM_WORKERS

    return SparseTextEmbedding(
        model_name=settings.SPARSE_MODEL_NAME,
        cache_dir=settings.MODELS_DIR,
        session_options=session_options,
        providers=settings.SPARSE_PROVIDERS,
        threads=settings.NUM_WORKERS,
    )
