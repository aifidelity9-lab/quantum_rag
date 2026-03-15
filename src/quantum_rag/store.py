from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.save(path, matrix)


def load_matrix(path: Path) -> np.ndarray:
    return np.load(path)
