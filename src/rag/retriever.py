from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.config.defaults import (
    CONTRACTNLI_ORIG_INDEX_PATH,
    CONTRACTNLI_ORIG_META_PATH,
    CONTRACTNLI_INDEX_PATH,
    CONTRACTNLI_META_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
    HYBRID_BM25_TOP_K,
    HYBRID_DENSE_TOP_K,
    HYBRID_RRF_K,
    HOTPOTQA_INDEX_PATH,
    HOTPOTQA_META_PATH,
    RERANK_CANDIDATE_K,
    RERANK_ENABLED,
    RERANK_MODEL_NAME,
    RETRIEVAL_MODE,
)
from src.indexing import (
    EmbeddingModel,
    build_bm25_index,
    load_faiss_index,
    load_metadata,
    search_bm25,
    search_index,
)

DATASET_CHOICES = ("legalbench_contractnli", "legalbench_mini", "contractnli_original", "hotpotqa")


@dataclass(frozen=True)
class RetrievedChunk:
    doc_path: str
    start: int
    end: int
    text: str
    score: float


class Retriever:
    def __init__(
        self,
        index_path=CONTRACTNLI_INDEX_PATH,
        meta_path=CONTRACTNLI_META_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        normalize=EMBEDDING_NORMALIZE,
        retrieval_mode: str = RETRIEVAL_MODE,
        hybrid_dense_top_k: int = HYBRID_DENSE_TOP_K,
        hybrid_bm25_top_k: int = HYBRID_BM25_TOP_K,
        hybrid_rrf_k: int = HYBRID_RRF_K,
        rerank_enabled: bool = RERANK_ENABLED,
        rerank_model_name: str = RERANK_MODEL_NAME,
        rerank_candidate_k: int = RERANK_CANDIDATE_K,
    ):
        self.index = load_faiss_index(index_path)
        self.metadata = load_metadata(meta_path)
        self.embedder = EmbeddingModel(embedding_model_name, normalize=normalize)
        self.retrieval_mode = retrieval_mode
        self.hybrid_dense_top_k = hybrid_dense_top_k
        self.hybrid_bm25_top_k = hybrid_bm25_top_k
        self.hybrid_rrf_k = hybrid_rrf_k
        self.rerank_enabled = rerank_enabled
        self.rerank_candidate_k = rerank_candidate_k
        self.bm25_index = None
        self.reranker = None
        if self.retrieval_mode == "hybrid":
            self.bm25_index = build_bm25_index(meta.get("text", "") for meta in self.metadata)
        self.doc_to_indices: dict[str, list[int]] = {}
        for idx, meta in enumerate(self.metadata):
            doc_path = meta.get("doc_path") or meta.get("file_name") or str(meta.get("doc_id"))
            self.doc_to_indices.setdefault(doc_path, []).append(idx)
        if self.rerank_enabled:
            from src.rag.reranker import CrossEncoderReranker

            self.reranker = CrossEncoderReranker(rerank_model_name)

    def _chunk_from_meta(self, idx: int, score: float) -> RetrievedChunk:
        meta = self.metadata[idx]
        doc_path = meta.get("doc_path") or meta.get("file_name") or str(meta.get("doc_id"))
        return RetrievedChunk(
            doc_path=doc_path,
            start=meta["start"],
            end=meta["end"],
            text=meta["text"],
            score=score,
        )

    def _dense_retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        vector = self.embedder.encode([query])[0]
        idxs, scores = search_index(self.index, vector, k)
        results: List[RetrievedChunk] = []
        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            results.append(self._chunk_from_meta(idx, float(score)))
        return results

    def _dense_retrieve_in_docs(self, query: str, k: int, doc_paths: List[str]) -> List[RetrievedChunk]:
        allowed: list[int] = []
        for doc_path in doc_paths:
            allowed.extend(self.doc_to_indices.get(doc_path, []))
        if not allowed:
            return []
        vector = np.array(self.embedder.encode([query])[0], dtype="float32")
        ranked: list[tuple[float, int]] = []
        for idx in allowed:
            chunk_vec = np.array(self.index.reconstruct(int(idx)), dtype="float32")
            score = float(np.dot(vector, chunk_vec))
            ranked.append((score, idx))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [self._chunk_from_meta(idx, score) for score, idx in ranked[:k]]

    def _bm25_retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        if self.bm25_index is None:
            return []
        idxs, scores = search_bm25(self.bm25_index, query, k)
        return [self._chunk_from_meta(idx, float(score)) for idx, score in zip(idxs, scores)]

    def _bm25_retrieve_in_docs(self, query: str, k: int, doc_paths: List[str]) -> List[RetrievedChunk]:
        allowed: list[int] = []
        for doc_path in doc_paths:
            allowed.extend(self.doc_to_indices.get(doc_path, []))
        if not allowed:
            return []
        texts = [self.metadata[idx].get("text", "") for idx in allowed]
        subset_index = build_bm25_index(texts)
        subset_idxs, scores = search_bm25(subset_index, query, k)
        mapped = [allowed[idx] for idx in subset_idxs]
        return [self._chunk_from_meta(idx, float(score)) for idx, score in zip(mapped, scores)]

    def _hybrid_retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        dense = self._dense_retrieve(query, max(k, self.hybrid_dense_top_k))
        bm25 = self._bm25_retrieve(query, max(k, self.hybrid_bm25_top_k))

        fused_scores: dict[tuple[str, int, int], float] = {}
        fused_chunks: dict[tuple[str, int, int], RetrievedChunk] = {}
        for ranked in (dense, bm25):
            for rank, chunk in enumerate(ranked, start=1):
                key = (chunk.doc_path, chunk.start, chunk.end)
                fused_chunks[key] = chunk
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (self.hybrid_rrf_k + rank)

        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)[:k]
        return [
            RetrievedChunk(
                doc_path=fused_chunks[key].doc_path,
                start=fused_chunks[key].start,
                end=fused_chunks[key].end,
                text=fused_chunks[key].text,
                score=fused_scores[key],
            )
            for key in ranked_keys
        ]

    def _hybrid_retrieve_in_docs(self, query: str, k: int, doc_paths: List[str]) -> List[RetrievedChunk]:
        dense = self._dense_retrieve_in_docs(query, max(k, self.hybrid_dense_top_k), doc_paths)
        bm25 = self._bm25_retrieve_in_docs(query, max(k, self.hybrid_bm25_top_k), doc_paths)

        fused_scores: dict[tuple[str, int, int], float] = {}
        fused_chunks: dict[tuple[str, int, int], RetrievedChunk] = {}
        for ranked in (dense, bm25):
            for rank, chunk in enumerate(ranked, start=1):
                key = (chunk.doc_path, chunk.start, chunk.end)
                fused_chunks[key] = chunk
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (self.hybrid_rrf_k + rank)

        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)[:k]
        return [
            RetrievedChunk(
                doc_path=fused_chunks[key].doc_path,
                start=fused_chunks[key].start,
                end=fused_chunks[key].end,
                text=fused_chunks[key].text,
                score=fused_scores[key],
            )
            for key in ranked_keys
        ]

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        candidate_k = max(k, self.rerank_candidate_k) if self.rerank_enabled else k
        if self.retrieval_mode == "dense":
            results = self._dense_retrieve(query, candidate_k)
        elif self.retrieval_mode == "hybrid":
            results = self._hybrid_retrieve(query, candidate_k)
        else:
            raise ValueError(f"Unknown retrieval_mode '{self.retrieval_mode}'")

        if self.rerank_enabled and self.reranker is not None:
            return self.reranker.rerank(query, results, top_k=k)

        return results[:k]

    def retrieve_in_docs(self, query: str, k: int, doc_paths: List[str]) -> List[RetrievedChunk]:
        candidate_k = max(k, self.rerank_candidate_k) if self.rerank_enabled else k
        if self.retrieval_mode == "dense":
            results = self._dense_retrieve_in_docs(query, candidate_k, doc_paths)
        elif self.retrieval_mode == "hybrid":
            results = self._hybrid_retrieve_in_docs(query, candidate_k, doc_paths)
        else:
            raise ValueError(f"Unknown retrieval_mode '{self.retrieval_mode}'")

        if self.rerank_enabled and self.reranker is not None:
            return self.reranker.rerank(query, results, top_k=k)

        return results[:k]


def resolve_dataset_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "legalbench_contractnli":
        return CONTRACTNLI_INDEX_PATH, CONTRACTNLI_META_PATH
    if dataset == "legalbench_mini":
        return CONTRACTNLI_INDEX_PATH, CONTRACTNLI_META_PATH
    if dataset == "contractnli_original":
        return CONTRACTNLI_ORIG_INDEX_PATH, CONTRACTNLI_ORIG_META_PATH
    if dataset == "hotpotqa":
        return HOTPOTQA_INDEX_PATH, HOTPOTQA_META_PATH
    raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {', '.join(DATASET_CHOICES)}")
