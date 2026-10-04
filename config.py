"""
Central configuration for RepoMind.
Loads everything from environment variables (.env) so no secrets are hardcoded.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    vector_db_dir: str = os.getenv("VECTOR_DB_DIR", "data/vector_db")
    repos_dir: str = os.getenv("REPOS_DIR", "data/repos")

    top_k: int = int(os.getenv("TOP_K", 6))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1200))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # File extensions RepoMind will index. Extend as needed.
    code_extensions: tuple = (
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".sql",
        ".md", ".yaml", ".yml", ".json", ".toml", ".sh",
    )

    max_file_size_bytes: int = 1_000_000  # skip files bigger than ~1MB


settings = Settings()

if not settings.openai_api_key:
    # Don't crash on import (useful for tests / MCP tool listing),
    # but every module that needs the key will raise a clear error when called.
    pass
