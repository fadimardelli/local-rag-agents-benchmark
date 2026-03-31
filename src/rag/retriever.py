from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
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

DATASET_CHOICES = ("legalbench_contractnli", "contractnli_original", "hotpotqa")


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
        self.doc_to_indices: dict[str, List[int]] = defaultdict(list)
        self.doc_name_to_paths: dict[str, List[str]] = defaultdict(list)
        for idx, meta in enumerate(self.metadata):
            doc_path = meta.get("doc_path") or meta.get("file_name") or str(meta.get("doc_id"))
            self.doc_to_indices[doc_path].append(idx)
            normalized_name = self._normalize_doc_name(doc_path)
            if doc_path not in self.doc_name_to_paths[normalized_name]:
                self.doc_name_to_paths[normalized_name].append(doc_path)
        for doc_indices in self.doc_to_indices.values():
            doc_indices.sort(key=lambda meta_idx: self.metadata[meta_idx]["start"])
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

    @staticmethod
    def _normalize_doc_name(doc_name: str) -> str:
        return Path(doc_name.strip()).name.strip().lower()

    def resolve_doc_path(self, doc_name: str) -> Optional[str]:
        normalized = self._normalize_doc_name(doc_name)
        matches = self.doc_name_to_paths.get(normalized, [])
        if not matches:
            return None
        return matches[0]

    def _dense_retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        vector = self.embedder.encode([query])[0]
        idxs, scores = search_index(self.index, vector, k)
        results: List[RetrievedChunk] = []
        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            results.append(self._chunk_from_meta(idx, float(score)))
        return results

    def _bm25_retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        if self.bm25_index is None:
            return []
        idxs, scores = search_bm25(self.bm25_index, query, k)
        return [self._chunk_from_meta(idx, float(score)) for idx, score in zip(idxs, scores)]

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

    def retrieve_in_document(self, query: str, doc_path: str, k: int) -> List[RetrievedChunk]:
        if k <= 0:
            return []

        doc_indices = self.doc_to_indices.get(doc_path, [])
        if not doc_indices:
            return self.retrieve(query, k)

        vector = np.array(self.embedder.encode([query])[0], dtype="float32")
        scored: List[tuple[int, float]] = []
        for meta_idx in doc_indices:
            chunk_vector = np.array(self.index.reconstruct(int(meta_idx)), dtype="float32")
            score = float(np.dot(vector, chunk_vector))
            scored.append((meta_idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [self._chunk_from_meta(meta_idx, score) for meta_idx, score in scored[:k]]

    def retrieve_diverse(self, query: str, k: int, *, candidate_k: Optional[int] = None, max_per_doc: int = 2) -> List[RetrievedChunk]:
        if k <= 0:
            return []

        expanded_k = max(candidate_k or (k * 5), k)
        if self.retrieval_mode == "dense":
            candidates = self._dense_retrieve(query, expanded_k)
        elif self.retrieval_mode == "hybrid":
            candidates = self._hybrid_retrieve(query, expanded_k)
        else:
            raise ValueError(f"Unknown retrieval_mode '{self.retrieval_mode}'")

        if self.rerank_enabled and self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=expanded_k)

        selected: List[RetrievedChunk] = []
        per_doc_counts: dict[str, int] = {}
        seen_keys: set[tuple[str, int, int]] = set()

        for chunk in candidates:
            key = (chunk.doc_path, chunk.start, chunk.end)
            if key in seen_keys:
                continue
            if per_doc_counts.get(chunk.doc_path, 0) >= max_per_doc:
                continue
            selected.append(chunk)
            seen_keys.add(key)
            per_doc_counts[chunk.doc_path] = per_doc_counts.get(chunk.doc_path, 0) + 1
            if len(selected) >= k:
                return selected

        for chunk in candidates:
            key = (chunk.doc_path, chunk.start, chunk.end)
            if key in seen_keys:
                continue
            selected.append(chunk)
            seen_keys.add(key)
            if len(selected) >= k:
                break

        return selected[:k]

    def expand_document(
        self,
        seed_chunks: List[RetrievedChunk],
        *,
        k: int,
        anchor: Optional[RetrievedChunk] = None,
        covered_positions: Optional[set[int]] = None,
    ) -> List[RetrievedChunk]:
        if not seed_chunks or k <= 0:
            return []

        anchor_chunk = anchor or seed_chunks[0]
        doc_indices = self.doc_to_indices.get(anchor_chunk.doc_path, [])
        if not doc_indices:
            return list(seed_chunks[:k])

        anchor_pos = self._find_doc_position(anchor_chunk.doc_path, anchor_chunk.start, anchor_chunk.end)
        if anchor_pos is None:
            return list(seed_chunks[:k])

        observed_positions = set(self.positions_for_doc_chunks(seed_chunks, anchor_chunk.doc_path))
        covered = set(covered_positions or set()) | observed_positions | {anchor_pos}

        left = min(covered)
        right = max(covered)
        new_positions: List[int] = []
        step = 1
        max_new = max(1, min(k, len(doc_indices) - len(covered)))
        while len(new_positions) < max_new and (left - step >= 0 or right + step < len(doc_indices)):
            right_candidate = right + step
            if right_candidate < len(doc_indices) and right_candidate not in covered:
                new_positions.append(right_candidate)
                if len(new_positions) >= max_new:
                    break
            left_candidate = left - step
            if left_candidate >= 0 and left_candidate not in covered:
                new_positions.append(left_candidate)
                if len(new_positions) >= max_new:
                    break
            step += 1

        # Expansion should augment the current evidence set, not discard it.
        # Preserve a portion of the strongest existing seed positions, then add
        # unseen frontier positions around the observed window.
        preserved_positions: List[int] = []
        preserve_budget = max(1, min(k // 2, len(observed_positions))) if observed_positions else 0
        preserved_rank: dict[int, int] = {}
        for chunk in seed_chunks:
            if chunk.doc_path != anchor_chunk.doc_path:
                continue
            pos = self._find_doc_position(chunk.doc_path, chunk.start, chunk.end)
            if pos is None or pos in preserved_positions:
                continue
            preserved_positions.append(pos)
            preserved_rank[pos] = len(preserved_positions)
            if len(preserved_positions) >= preserve_budget:
                break

        new_position_set = set(new_positions)
        preserved_position_set = set(preserved_positions)

        selected_positions = self._select_progressive_window(
            anchor_pos=anchor_pos,
            covered_positions=covered,
            new_positions=new_positions,
            preserved_positions=preserved_positions,
            k=k,
        )

        results = [
            self._chunk_from_meta(
                doc_indices[pos],
                score=self._progressive_expansion_score(
                    anchor_pos,
                    pos,
                    is_new=pos in new_position_set,
                    is_preserved=pos in preserved_position_set,
                    preserved_rank=preserved_rank.get(pos),
                ),
            )
            for pos in selected_positions
        ]
        results.sort(key=lambda chunk: chunk.score, reverse=True)
        return results[:k]

    def focus_document(
        self,
        seed_chunks: List[RetrievedChunk],
        *,
        doc_path: str,
        k: int,
        anchor: Optional[RetrievedChunk] = None,
    ) -> List[RetrievedChunk]:
        if k <= 0:
            return []

        doc_indices = self.doc_to_indices.get(doc_path, [])
        if not doc_indices:
            return []

        doc_seed_positions = self.positions_for_doc_chunks(seed_chunks, doc_path)
        if anchor is not None and anchor.doc_path == doc_path:
            anchor_pos = self._find_doc_position(anchor.doc_path, anchor.start, anchor.end)
        elif doc_seed_positions:
            anchor_pos = doc_seed_positions[0]
        else:
            anchor_pos = 0

        if anchor_pos is None:
            anchor_pos = 0

        selected_positions: List[int] = []
        for pos in sorted(doc_seed_positions, key=lambda pos: (abs(pos - anchor_pos), pos)):
            if pos not in selected_positions:
                selected_positions.append(pos)
            if len(selected_positions) >= k:
                break

        radius = 1
        while len(selected_positions) < k and (anchor_pos - radius >= 0 or anchor_pos + radius < len(doc_indices)):
            right = anchor_pos + radius
            if right < len(doc_indices) and right not in selected_positions:
                selected_positions.append(right)
                if len(selected_positions) >= k:
                    break
            left = anchor_pos - radius
            if left >= 0 and left not in selected_positions:
                selected_positions.append(left)
                if len(selected_positions) >= k:
                    break
            radius += 1

        results = [
            self._chunk_from_meta(
                doc_indices[pos],
                score=self._focus_score(anchor_pos, pos, is_seed=(pos in doc_seed_positions)),
            )
            for pos in selected_positions
        ]
        results.sort(key=lambda chunk: chunk.score, reverse=True)
        return results[:k]

    def positions_for_doc_chunks(self, chunks: List[RetrievedChunk], doc_path: str) -> List[int]:
        positions: List[int] = []
        for chunk in chunks:
            if chunk.doc_path != doc_path:
                continue
            pos = self._find_doc_position(chunk.doc_path, chunk.start, chunk.end)
            if pos is not None and pos not in positions:
                positions.append(pos)
        return positions

    def has_unseen_document_frontier(
        self,
        seed_chunks: List[RetrievedChunk],
        *,
        anchor: Optional[RetrievedChunk] = None,
        covered_positions: Optional[set[int]] = None,
    ) -> bool:
        if not seed_chunks:
            return False

        anchor_chunk = anchor or seed_chunks[0]
        doc_indices = self.doc_to_indices.get(anchor_chunk.doc_path, [])
        if not doc_indices:
            return False

        anchor_pos = self._find_doc_position(anchor_chunk.doc_path, anchor_chunk.start, anchor_chunk.end)
        if anchor_pos is None:
            return False

        observed_positions = set(self.positions_for_doc_chunks(seed_chunks, anchor_chunk.doc_path))
        covered = set(covered_positions or set()) | observed_positions | {anchor_pos}
        if len(covered) >= len(doc_indices):
            return False

        left = min(covered)
        right = max(covered)
        return left > 0 or right < len(doc_indices) - 1

    @staticmethod
    def _select_progressive_window(
        *,
        anchor_pos: int,
        covered_positions: set[int],
        new_positions: List[int],
        preserved_positions: List[int],
        k: int,
    ) -> List[int]:
        new_set = set(new_positions)
        selected: List[int] = []

        for pos in preserved_positions:
            if pos not in selected:
                selected.append(pos)
            if len(selected) >= k:
                return selected[:k]

        for pos in new_positions:
            if pos not in selected:
                selected.append(pos)
            if len(selected) >= k:
                return selected[:k]

        remaining = [pos for pos in covered_positions if pos not in new_set and pos not in set(preserved_positions)]
        remaining.sort(key=lambda pos: (abs(pos - anchor_pos), pos))
        for pos in remaining:
            if len(selected) >= k:
                break
            selected.append(pos)

        if not selected:
            selected.append(anchor_pos)

        return selected[:k]

    def _find_doc_position(self, doc_path: str, start: int, end: int) -> Optional[int]:
        doc_indices = self.doc_to_indices.get(doc_path, [])
        for pos, meta_idx in enumerate(doc_indices):
            meta = self.metadata[meta_idx]
            if int(meta["start"]) == int(start) and int(meta["end"]) == int(end):
                return pos
        return None

    @staticmethod
    def _progressive_expansion_score(
        anchor_pos: int,
        pos: int,
        *,
        is_new: bool,
        is_preserved: bool,
        preserved_rank: Optional[int],
    ) -> float:
        base = 1.0 / (1 + abs(pos - anchor_pos))
        if is_preserved:
            rank_bonus = 1.0 / max(1, preserved_rank or 1)
            return 2.0 + rank_bonus + base
        if is_new:
            return 1.0 + base
        return base

    @staticmethod
    def _focus_score(anchor_pos: int, pos: int, *, is_seed: bool) -> float:
        base = 1.0 / (1 + abs(pos - anchor_pos))
        return base + (0.5 if is_seed else 0.0)


def resolve_dataset_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "legalbench_contractnli":
        return CONTRACTNLI_INDEX_PATH, CONTRACTNLI_META_PATH
    if dataset == "contractnli_original":
        return CONTRACTNLI_ORIG_INDEX_PATH, CONTRACTNLI_ORIG_META_PATH
    if dataset == "hotpotqa":
        return HOTPOTQA_INDEX_PATH, HOTPOTQA_META_PATH
    raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {', '.join(DATASET_CHOICES)}")
