"""
Splits file contents into overlapping, token-bounded chunks.
Chunking is line-based (not naive character slicing) so a chunk never
starts or ends mid-line, which keeps code readable in the LLM's context.
"""
from dataclasses import dataclass

import tiktoken

from config import settings
from src.ingestion.repo_loader import RawFile

_encoder = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    id: str
    repo_name: str
    file_path: str
    start_line: int
    end_line: int
    text: str


def _token_len(text: str) -> int:
    return len(_encoder.encode(text))


def chunk_file(raw: RawFile, chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    lines = raw.content.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    start_idx = 0
    chunk_num = 0

    while start_idx < len(lines):
        current_lines: list[str] = []
        tokens_so_far = 0
        idx = start_idx

        while idx < len(lines):
            line_tokens = _token_len(lines[idx]) + 1
            if current_lines and tokens_so_far + line_tokens > chunk_size:
                break
            current_lines.append(lines[idx])
            tokens_so_far += line_tokens
            idx += 1

        if not current_lines:
            # single line longer than chunk_size; take it anyway to avoid infinite loop
            current_lines = [lines[start_idx]]
            idx = start_idx + 1

        text = "\n".join(current_lines)
        chunks.append(
            Chunk(
                id=f"{raw.repo_name}::{raw.path}::{chunk_num}",
                repo_name=raw.repo_name,
                file_path=raw.path,
                start_line=start_idx + 1,
                end_line=idx,
                text=text,
            )
        )
        chunk_num += 1

        if idx >= len(lines):
            break

        # step back by `overlap` tokens' worth of lines for context continuity
        overlap_lines = 0
        overlap_tokens = 0
        back_idx = idx - 1
        while back_idx > start_idx and overlap_tokens < overlap:
            overlap_tokens += _token_len(lines[back_idx]) + 1
            overlap_lines += 1
            back_idx -= 1
        start_idx = max(start_idx + 1, idx - overlap_lines)

    return chunks


def chunk_files(raw_files: list[RawFile]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for rf in raw_files:
        all_chunks.extend(chunk_file(rf))
    return all_chunks
