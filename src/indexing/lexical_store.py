from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class BM25Index:
    vectorizer: CountVectorizer
    term_matrix_csc: any
    doc_len: np.ndarray
    avgdl: float
    idf: np.ndarray
    k1: float = 1.5
    b: float = 0.75


def build_bm25_index(
    texts: Iterable[str],
    *,
    lowercase: bool = True,
    token_pattern: str = r"(?u)\b\w+\b",
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    docs = list(texts)
    if not docs:
        raise ValueError("No texts provided to build BM25 index")

    vectorizer = CountVectorizer(lowercase=lowercase, token_pattern=token_pattern)
    term_matrix_csr = vectorizer.fit_transform(docs).astype(np.float32)
    doc_len = np.asarray(term_matrix_csr.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_len.mean()) if len(doc_len) else 0.0

    n_docs = term_matrix_csr.shape[0]
    df = np.diff(term_matrix_csr.tocsc().indptr).astype(np.float32)
    # BM25 idf variant commonly used in practice; clip at 0 to avoid negative weights.
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    return BM25Index(
        vectorizer=vectorizer,
        term_matrix_csc=term_matrix_csr.tocsc(),
        doc_len=doc_len,
        avgdl=avgdl,
        idf=idf,
        k1=k1,
        b=b,
    )


def search_bm25(index: BM25Index, query: str, k: int) -> Tuple[List[int], List[float]]:
    if k <= 0:
        raise ValueError("k must be > 0")

    q = index.vectorizer.transform([query])
    if q.nnz == 0:
        return [], []

    scores: dict[int, float] = {}
    for term_idx, qtf in zip(q.indices, q.data):
        col_start = index.term_matrix_csc.indptr[term_idx]
        col_end = index.term_matrix_csc.indptr[term_idx + 1]
        doc_ids = index.term_matrix_csc.indices[col_start:col_end]
        term_freqs = index.term_matrix_csc.data[col_start:col_end]
        if len(doc_ids) == 0:
            continue
        denom = term_freqs + index.k1 * (1.0 - index.b + index.b * index.doc_len[doc_ids] / max(index.avgdl, 1e-6))
        contrib = index.idf[term_idx] * (term_freqs * (index.k1 + 1.0) / denom) * qtf
        for doc_id, value in zip(doc_ids.tolist(), contrib.tolist()):
            scores[doc_id] = scores.get(doc_id, 0.0) + float(value)

    if not scores:
        return [], []

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    idxs = [doc_id for doc_id, _ in ranked]
    vals = [score for _, score in ranked]
    return idxs, vals
