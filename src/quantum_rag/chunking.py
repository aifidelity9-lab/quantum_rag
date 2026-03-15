from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    page: int | None


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def split_into_chunks(
    pages: Iterable[tuple[int | None, str]],
    source: str,
    chunk_size: int = 900,
    overlap: int = 180,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_id = 0

    for page, raw_text in pages:
        text = normalize_text(raw_text)
        if not text:
            continue

        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=source,
                        page=page,
                    )
                )
                chunk_id += 1

            if end >= text_len:
                break

            start = max(end - overlap, start + 1)

    return chunks
