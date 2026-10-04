"""
Exposes RepoMind's RAG pipeline as MCP tools, so any MCP-compatible client
(Claude Desktop, Claude Code, etc.) can index a repo and ask questions
about it directly, without going through the Streamlit UI.

Run standalone with:
    python -m src.mcp_server.server
"""
from mcp.server.fastmcp import FastMCP

from src.rag import pipeline
from src.vectorstore import chroma_store

mcp = FastMCP("RepoMind")


@mcp.tool()
def index_repo(path_or_url: str) -> dict:
    """
    Index a codebase for RAG search. Accepts a local directory path or a
    GitHub repo URL. Must be called once before search_codebase / ask_codebase
    can answer questions about a given repo.
    """
    return pipeline.index_source(path_or_url)


@mcp.tool()
def list_indexed_repos() -> list[str]:
    """List the names of all repos currently indexed and searchable."""
    return pipeline.list_indexed_repos()


@mcp.tool()
def search_codebase(repo_name: str, query: str, top_k: int = 6) -> list[dict]:
    """
    Semantic search over an indexed repo's code. Returns the top matching
    chunks with file path, line range, and the code text itself -
    without generating a natural-language answer.
    """
    hits = chroma_store.query(repo_name, query, top_k=top_k)
    return [
        {
            "file_path": h["metadata"]["file_path"],
            "start_line": h["metadata"]["start_line"],
            "end_line": h["metadata"]["end_line"],
            "text": h["text"],
        }
        for h in hits
    ]


@mcp.tool()
def ask_codebase(repo_name: str, question: str, top_k: int = 6) -> dict:
    """
    Ask a natural-language question about an indexed repo. Retrieves the
    most relevant code chunks and generates a grounded answer with cited
    file paths and line ranges.
    """
    return pipeline.ask(repo_name, question, top_k=top_k)


@mcp.tool()
def get_file_snippet(repo_name: str, file_path: str, start_line: int, end_line: int) -> str:
    """
    Fetch the raw text of a specific line range from an already-indexed
    repo's chunk store (useful for expanding context around a search hit).
    """
    hits = chroma_store.query(repo_name, file_path, top_k=20)
    for h in hits:
        m = h["metadata"]
        if m["file_path"] == file_path and m["start_line"] <= start_line and m["end_line"] >= end_line:
            return h["text"]
    return "Snippet not found in indexed chunks; try search_codebase instead."


if __name__ == "__main__":
    mcp.run()
