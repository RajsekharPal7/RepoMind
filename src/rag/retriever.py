"""
High-level retrieval: given a natural-language question, fetch the most
relevant code chunks from the indexed repo and format them for the LLM.
"""
from dataclasses import dataclass

from src.vectorstore import chroma_store


@dataclass
class RetrievedChunk:
    file_path: str
    start_line: int
    end_line: int
    text: str
    distance: float


def retrieve(repo_name: str, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    hits = chroma_store.query(repo_name, question, top_k=top_k)
    return [
        RetrievedChunk(
            file_path=h["metadata"]["file_path"],
            start_line=h["metadata"]["start_line"],
            end_line=h["metadata"]["end_line"],
            text=h["text"],
            distance=h["distance"],
        )
        for h in hits
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as labeled code blocks for the prompt."""
    blocks = []
    for c in chunks:
        header = f"# File: {c.file_path} (lines {c.start_line}-{c.end_line})"
        blocks.append(f"{header}\n```\n{c.text}\n```")
    return "\n\n".join(blocks)
