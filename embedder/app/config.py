import os


class EmbedderSettings:
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    COHERE_EMBED_MODEL: str = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-light-v3.0")
    COHERE_INPUT_TYPE: str = os.getenv("COHERE_EMBED_INPUT_TYPE", "search_document")
    COHERE_RETRIES: int = int(os.getenv("COHERE_EMBED_RETRIES", "3"))
    COHERE_RETRY_DELAY: int = int(os.getenv("COHERE_EMBED_RETRY_DELAY", "5"))
    COHERE_PRICE_PER_1M_TOKENS: float = float(os.getenv("EMBEDDER_COHERE_PRICE_PER_1M", "0.10"))
    COHERE_MAX_TOKENS: int = int(os.getenv("EMBEDDER_COHERE_MAX_TOKENS", "512"))

    ES_HOST: str = os.getenv("EMBEDDER_ES_HOST", "http://elasticsearch:9200")
    ES_INDEX: str = os.getenv("EMBEDDER_ES_INDEX", "jodal_documents")
    ES_TIMEOUT: int = int(os.getenv("EMBEDDER_ES_TIMEOUT", "30"))
    ES_SCROLL_TIME: str = os.getenv("EMBEDDER_ES_SCROLL_TIME", "30m")
    ES_MAX_RETRIES: int = int(os.getenv("EMBEDDER_ES_MAX_RETRIES", "3"))

    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT", "60"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "nederland_cohere")
    QDRANT_UPSERT_BATCH_SIZE: int = int(os.getenv("EMBEDDER_QDRANT_UPSERT_BATCH_SIZE", "5000"))

    MAX_DOCUMENTS_TO_PROCESS: int = int(os.getenv("EMBEDDER_MAX_DOCUMENTS", "-1"))
    BATCH_SIZE: int = int(os.getenv("EMBEDDER_BATCH_SIZE", "50"))
    PROCESSING_THRESHOLD_MULTIPLIER: int = int(os.getenv("EMBEDDER_THRESHOLD_MULTIPLIER", "5"))

    ESTIMATE_SAMPLE_SIZE: int = int(os.getenv("EMBEDDER_ESTIMATE_SAMPLE_SIZE", "500"))
    ESTIMATE_SEED: int = int(os.getenv("EMBEDDER_ESTIMATE_SEED", "42"))
    ESTIMATE_BOOTSTRAP_ITERATIONS: int = int(os.getenv("EMBEDDER_ESTIMATE_BOOTSTRAP_ITERATIONS", "1000"))
    ESTIMATE_CHARS_PER_TOKEN: float = float(os.getenv("EMBEDDER_ESTIMATE_CHARS_PER_TOKEN", "3.5"))
    ESTIMATE_SOURCE_FIELD: str = os.getenv("EMBEDDER_ESTIMATE_SOURCE_FIELD", "source")

    DENSE_BATCH_SIZE: int = int(os.getenv("EMBEDDER_DENSE_BATCH_SIZE", "96"))
    SPARSE_BATCH_SIZE: int = int(os.getenv("EMBEDDER_SPARSE_BATCH_SIZE", "1000"))

    COMBINE_TEXT_UNDER_N_CHARS: int = int(os.getenv("EMBEDDER_COMBINE_UNDER", "800"))
    MAX_CHARACTERS: int = int(os.getenv("EMBEDDER_MAX_CHARS", "1000"))
    NEW_AFTER_N_CHARS: int = int(os.getenv("EMBEDDER_NEW_AFTER", "1000"))
    MAX_PARTITION: int = int(os.getenv("EMBEDDER_MAX_PARTITION", "1000"))
    OVERLAP: int = int(os.getenv("EMBEDDER_OVERLAP", "75"))

    SPARSE_MODEL_NAME: str = os.getenv("EMBEDDER_SPARSE_MODEL", "Qdrant/bm25")
    SPARSE_PROVIDERS: list = ["CPUExecutionProvider"]

    MODELS_DIR: str = os.getenv("EMBEDDER_MODELS_DIR", "/app/embedder/models")
    STATE_FILE: str = os.getenv("EMBEDDER_STATE_FILE", "/data/embedder_state/last_run.txt")

    CPU_COUNT: int = os.cpu_count() or 1

    @property
    def NUM_WORKERS(self) -> int:
        return max(1, self.CPU_COUNT - 2)


settings = EmbedderSettings()
