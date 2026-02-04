import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np


def build_faiss_index(vectors: List[List[float]], normalize: bool = True) -> faiss.Index:
    if not vectors:
        raise ValueError("No vectors provided to build index")
    mat = np.array(vectors, dtype="float32")
    dim = mat.shape[1]
    if normalize:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(mat)
    return index


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def save_metadata(metadata: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_faiss_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    return faiss.read_index(str(path))


def load_metadata(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def search_index(index: faiss.Index, query_vector: List[float], k: int) -> Tuple[List[int], List[float]]:
    if k <= 0:
        raise ValueError("k must be > 0")
    q = np.array([query_vector], dtype="float32")
    scores, idxs = index.search(q, k)
    return idxs[0].tolist(), scores[0].tolist()
