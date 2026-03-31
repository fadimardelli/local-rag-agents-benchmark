from typing import List, Optional

from sentence_transformers import CrossEncoder

from src.rag.retriever import RetrievedChunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(scores.tolist(), chunks),
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            RetrievedChunk(
                doc_path=chunk.doc_path,
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
                score=float(score),
            )
            for score, chunk in ranked[:top_k]
        ]
