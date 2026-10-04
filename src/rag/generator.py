"""
Builds the final RAG prompt and calls OpenAI's chat completions endpoint
(model = gpt-4.1 by default, configurable via OPENAI_CHAT_MODEL) to answer
questions about the codebase, grounded in retrieved chunks.
"""
from openai import OpenAI

from config import settings
from src.rag.retriever import RetrievedChunk, format_context
from src.utils.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """You are RepoMind, an expert assistant that answers questions about a specific \
codebase using ONLY the provided context chunks (retrieved via RAG from the repo).

Rules:
- Ground every claim in the provided context. If the context doesn't contain the answer, say so plainly.
- Always cite the file path and line range for any code you reference, e.g. (src/app.py, lines 10-25).
- Prefer concise, technically precise answers. Use code blocks for code.
- If asked to explain a function/class, walk through its logic using the exact code shown.
- Never invent files, functions, or behavior that isn't in the context.
"""

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks) if chunks else "(no relevant context retrieved)"
    return (
        f"### Retrieved code context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        f"Answer the question using only the context above."
    )


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> str:
    """
    history: optional prior turns as [{"role": "user"/"assistant", "content": str}, ...]
    """
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": build_user_prompt(question, chunks)})

    log.info(f"Calling {settings.chat_model} with {len(chunks)} retrieved chunks")
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content
