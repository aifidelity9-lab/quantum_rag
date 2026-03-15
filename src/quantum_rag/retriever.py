from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .store import load_json, load_matrix
from .vectorizer import HashedTfidfVectorizer


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: int
    text: str
    source: str
    page: int | None

    def citation(self) -> str:
        if self.page is not None:
            return f"{self.source}, page {self.page}"
        return self.source


class RagIndex:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.metadata = load_json(dataset_dir / "metadata.json")
        self.matrix = load_matrix(dataset_dir / "matrix.npy")
        self.vectorizer = HashedTfidfVectorizer(
            dims=int(self.metadata["vectorizer"]["dims"]),
            min_token_len=int(self.metadata["vectorizer"]["min_token_len"]),
        )

    def search(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vec = self.vectorizer.transform([question], self.metadata["vectorizer"])[0]
        if np.linalg.norm(query_vec) == 0:
            return []

        scores = self.matrix @ query_vec
        if scores.size == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        chunks = self.metadata["chunks"]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            chunk = chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    score=float(scores[idx]),
                    chunk_id=int(chunk["chunk_id"]),
                    text=str(chunk["text"]),
                    source=str(chunk["source"]),
                    page=chunk["page"],
                )
            )
        return results


def answer_question(question: str, results: list[RetrievedChunk]) -> str:
    if not results:
        return "I could not find a grounded answer in the indexed book for that question."

    best = results[0]
    lead = best.text.strip().replace("\n", " ")
    if len(lead) > 420:
        lead = lead[:420].rsplit(" ", 1)[0] + "..."

    if best.page is not None:
        citation = f"{best.source}, page {best.page}"
    else:
        citation = best.source

    return (
        "Based on the most relevant passage, the book indicates: "
        f"{lead} [{citation}]"
    )


def format_context(results: list[RetrievedChunk], max_chars: int = 4000) -> str:
    parts: list[str] = []
    total = 0
    for rank, item in enumerate(results, start=1):
        block = (
            f"[Passage {rank} | {item.citation()} | score={item.score:.4f}]\n"
            f"{item.text.strip()}\n"
        )
        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)
