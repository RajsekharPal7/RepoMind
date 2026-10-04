"""
Top-level orchestration used by both the Streamlit UI and the MCP server,
so the two front-ends never duplicate logic.
"""
from src.ingestion.chunker import chunk_files
from src.ingestion.repo_loader import load_source
from src.rag.generator import answer_question
from src.rag.retriever import retrieve
from src.vectorstore import chroma_store


def index_source(path_or_url: str) -> dict:
    """Load, chunk, embed and store a repo (local path or GitHub URL)."""
    repo_name, raw_files = load_source(path_or_url)
    chunks = chunk_files(raw_files)
    num_indexed = chroma_store.index_chunks(repo_name, chunks)
    return {
        "repo_name": repo_name,
        "files_loaded": len(raw_files),
        "chunks_indexed": num_indexed,
    }


def list_indexed_repos() -> list[str]:
    return chroma_store.list_repos()


def ask(repo_name: str, question: str, top_k: int | None = None, history: list[dict] | None = None) -> dict:
    chunks = retrieve(repo_name, question, top_k=top_k)
    answer = answer_question(question, chunks, history=history)
    return {
        "answer": answer,
        "sources": [
            {"file_path": c.file_path, "start_line": c.start_line, "end_line": c.end_line}
            for c in chunks
        ],
    }
