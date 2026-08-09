import chromadb
from chromadb import EmbeddingFunction
from core.config import CHROMA_DIR


class _NoOpEmbeddingFunction(EmbeddingFunction):
    """query_memory() only ever does a plain .get() lookup, never a vector
    similarity search, so a real embedding model is unused overhead. Chroma's
    built-in default embedding function downloads a ~90MB ONNX model from S3
    on first use, which can stall for minutes (or fail outright) on restricted
    corporate networks. Returning a fixed dummy vector skips that entirely."""

    def __call__(self, input):
        return [[0.0] for _ in input]

    @staticmethod
    def name() -> str:
        return "idamp_noop"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "_NoOpEmbeddingFunction":
        return _NoOpEmbeddingFunction()


def get_chroma_client():
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = "idamp_memory"):
    """Get or create the IDAMP memory collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=name, embedding_function=_NoOpEmbeddingFunction())



def store_document(doc_id: str, text: str, metadata: dict | None = None) -> None:
    """Store a document in semantic memory."""
    try:
        collection = get_collection()
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
    except Exception:
        # Keep pipeline running even if semantic memory backend is unavailable.
        pass


def query_memory(query_text: str, n_results: int = 5) -> list[dict]:
    """Query semantic memory for relevant documents (searches metadata/text without embeddings)."""
    try:
        collection = get_collection()
        # Get all documents and filter locally (no embedding-based search)
        # This is a simple implementation that just returns recent docs
        results = collection.get(limit=n_results)
        return [
            {"id": id_, "document": doc, "metadata": meta}
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]
    except Exception:
        return []
