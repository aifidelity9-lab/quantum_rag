from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def hash_token(token: str, dims: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dims


@dataclass
class HashedTfidfVectorizer:
    dims: int = 2048
    min_token_len: int = 2

    def fit_transform(self, texts: list[str]) -> tuple[np.ndarray, dict[str, object]]:
        doc_freq = np.zeros(self.dims, dtype=np.float64)
        term_counts: list[dict[int, int]] = []

        for text in texts:
            counts: dict[int, int] = {}
            seen: set[int] = set()
            for token in tokenize(text):
                if len(token) < self.min_token_len:
                    continue
                idx = hash_token(token, self.dims)
                counts[idx] = counts.get(idx, 0) + 1
                seen.add(idx)
            term_counts.append(counts)
            for idx in seen:
                doc_freq[idx] += 1.0

        n_docs = max(len(texts), 1)
        idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0

        matrix = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, counts in enumerate(term_counts):
            if not counts:
                continue
            max_tf = max(counts.values())
            for idx, count in counts.items():
                tf = 0.5 + 0.5 * (count / max_tf)
                matrix[row, idx] = tf * idf[idx]
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm

        meta = {
            "dims": self.dims,
            "min_token_len": self.min_token_len,
            "idf": idf.tolist(),
        }
        return matrix, meta

    def transform(self, texts: list[str], meta: dict[str, object]) -> np.ndarray:
        dims = int(meta["dims"])
        idf = np.array(meta["idf"], dtype=np.float32)
        matrix = np.zeros((len(texts), dims), dtype=np.float32)

        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for token in tokenize(text):
                if len(token) < self.min_token_len:
                    continue
                idx = hash_token(token, dims)
                counts[idx] = counts.get(idx, 0) + 1

            if not counts:
                continue

            max_tf = max(counts.values())
            for idx, count in counts.items():
                tf = 0.5 + 0.5 * (count / max_tf)
                matrix[row, idx] = tf * idf[idx]

            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm

        return matrix
