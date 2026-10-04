from src.ingestion.chunker import chunk_file
from src.ingestion.repo_loader import RawFile


def test_chunk_file_basic():
    content = "\n".join([f"line {i}" for i in range(1, 51)])
    raw = RawFile(path="foo.py", abs_path="/tmp/foo.py", content=content, repo_name="demo")

    chunks = chunk_file(raw, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    # every chunk stays within the file's line range
    for c in chunks:
        assert 1 <= c.start_line <= c.end_line <= 50
    # chunks cover the whole file
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 50


def test_chunk_file_empty():
    raw = RawFile(path="empty.py", abs_path="/tmp/empty.py", content="", repo_name="demo")
    assert chunk_file(raw) == []
