"""
Thin wrapper around the OpenAI embeddings endpoint.
Batches requests to stay within API limits and keep indexing fast.
"""
from openai import OpenAI

from config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts, returning one vector per input text, order preserved."""
    if not texts:
        return []

    client = get_client()
    vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        log.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} chunks)")
        response = client.embeddings.create(model=settings.embed_model, input=batch)
        vectors.extend([item.embedding for item in response.data])

    return vectors


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
