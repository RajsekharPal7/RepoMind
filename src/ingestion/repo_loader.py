"""
Loads a codebase into memory: either clones a remote GitHub URL or reads
an already-local directory, then walks it and returns raw file records.
"""
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import git

from config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist",
    "build", ".next", ".idea", ".vscode", "target", "vendor",
}


@dataclass
class RawFile:
    path: str          # path relative to repo root
    abs_path: str       # absolute path on disk
    content: str
    repo_name: str


def _on_rm_error(func, path, exc_info):
    """Handle read-only files on Windows/CI clones."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repo(github_url: str, dest_dir: str | None = None) -> str:
    """Clone a GitHub repo (shallow) into REPOS_DIR and return the local path."""
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest_dir = dest_dir or os.path.join(settings.repos_dir, repo_name)

    if os.path.exists(dest_dir):
        log.info(f"Repo already cloned at {dest_dir}, pulling latest instead of re-cloning.")
        shutil.rmtree(dest_dir, onerror=_on_rm_error)

    os.makedirs(settings.repos_dir, exist_ok=True)
    log.info(f"Cloning {github_url} -> {dest_dir}")
    git.Repo.clone_from(github_url, dest_dir, depth=1)
    return dest_dir


def iter_source_files(root_dir: str) -> Iterator[RawFile]:
    """Walk a local directory and yield readable source files, skipping noise."""
    root = Path(root_dir).resolve()
    repo_name = root.name

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]

        for filename in filenames:
            abs_path = Path(dirpath) / filename
            if abs_path.suffix.lower() not in settings.code_extensions:
                continue
            try:
                size = abs_path.stat().st_size
                if size == 0 or size > settings.max_file_size_bytes:
                    continue
                text = abs_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError) as e:
                log.warning(f"Skipping unreadable file {abs_path}: {e}")
                continue

            rel_path = str(abs_path.relative_to(root))
            yield RawFile(path=rel_path, abs_path=str(abs_path), content=text, repo_name=repo_name)


def load_source(path_or_url: str) -> tuple[str, list[RawFile]]:
    """
    Accepts either a local directory path or a github URL.
    Returns (repo_name, list_of_raw_files).
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://") or path_or_url.endswith(".git"):
        local_dir = clone_repo(path_or_url)
    else:
        local_dir = path_or_url
        if not os.path.isdir(local_dir):
            raise ValueError(f"Local path does not exist or is not a directory: {local_dir}")

    files = list(iter_source_files(local_dir))
    repo_name = Path(local_dir).resolve().name
    log.info(f"Loaded {len(files)} source files from '{repo_name}'")
    return repo_name, files
