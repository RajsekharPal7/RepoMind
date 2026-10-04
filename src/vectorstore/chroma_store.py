"""
Persistent local vector store backed by Chroma.
One Chroma "collection" per indexed repo, so multiple repos can coexist.
"""
import chromadb

from config import settings
from src.embeddings.embedder import embed_query, embed_texts
from src.ingestion.chunker import Chunk
from src.utils.logger import get_logger

log = get_logger(__name__)

_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.vector_db_dir)
    return _client


def _collection_name(repo_name: str) -> str:
    # Chroma collection names must be simple; sanitize.
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in repo_name)[:60] or "repo"


def index_chunks(repo_name: str, chunks: list[Chunk]) -> int:
    """Embed and store all chunks for a repo. Replaces any existing collection for it."""
    client = get_client()
    name = _collection_name(repo_name)

    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name)

    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    collection.add(
        ids=[c.id for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[
            {
                "repo_name": c.repo_name,
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in chunks
        ],
    )
    log.info(f"Indexed {len(chunks)} chunks into collection '{name}'")
    return len(chunks)


def list_repos() -> list[str]:
    client = get_client()
    return [c.name for c in client.list_collections()]


def query(repo_name: str, question: str, top_k: int | None = None) -> list[dict]:
    """Return the top_k most relevant chunks for a question within one repo's collection."""
    client = get_client()
    name = _collection_name(repo_name)
    try:
        collection = client.get_collection(name)
    except Exception:
        raise ValueError(f"No indexed collection found for repo '{repo_name}'. Index it first.")

    top_k = top_k or settings.top_k
    q_vector = embed_query(question)
    results = collection.query(query_embeddings=[q_vector], n_results=top_k)

    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits
