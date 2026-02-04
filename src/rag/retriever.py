from dataclasses import dataclass
from typing import List

from src.config.defaults import (
    CONTRACTNLI_INDEX_PATH,
    CONTRACTNLI_META_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
)
from src.indexing import EmbeddingModel, load_faiss_index, load_metadata, search_index


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
    ):
        self.index = load_faiss_index(index_path)
        self.metadata = load_metadata(meta_path)
        self.embedder = EmbeddingModel(embedding_model_name, normalize=normalize)

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        vector = self.embedder.encode([query])[0]
        idxs, scores = search_index(self.index, vector, k)
        results: List[RetrievedChunk] = []
        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            results.append(
                RetrievedChunk(
                    doc_path=meta["doc_path"],
                    start=meta["start"],
                    end=meta["end"],
                    text=meta["text"],
                    score=score,
                )
            )
        return results
