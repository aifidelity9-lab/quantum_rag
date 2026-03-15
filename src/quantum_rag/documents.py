from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def load_document(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {suffix}")


def load_pdf(path: Path) -> list[tuple[int | None, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int | None, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((index, text))
    return pages
