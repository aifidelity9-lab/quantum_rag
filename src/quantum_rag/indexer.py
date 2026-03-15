from __future__ import annotations

from pathlib import Path

from .chunking import Chunk, split_into_chunks
from .documents import load_document
from .store import ensure_dir, save_json, save_matrix
from .vectorizer import HashedTfidfVectorizer


def build_index(
    input_path: Path,
    dataset_name: str,
    data_dir: Path,
    chunk_size: int = 900,
    overlap: int = 180,
    dims: int = 2048,
) -> Path:
    pages = load_document(input_path)
    chunks = split_into_chunks(
        pages=pages,
        source=input_path.name,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    vectorizer = HashedTfidfVectorizer(dims=dims)
    matrix, vector_meta = vectorizer.fit_transform([chunk.text for chunk in chunks])

    dataset_dir = data_dir / dataset_name
    ensure_dir(dataset_dir)

    metadata = {
        "dataset_name": dataset_name,
        "source_file": str(input_path),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "vectorizer": vector_meta,
        "chunks": [chunk_to_dict(chunk) for chunk in chunks],
    }

    save_json(dataset_dir / "metadata.json", metadata)
    save_matrix(dataset_dir / "matrix.npy", matrix)
    return dataset_dir


def chunk_to_dict(chunk: Chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "source": chunk.source,
        "page": chunk.page,
    }
