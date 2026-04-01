from typing import Iterable, List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str, normalize: bool = True, device: str | None = None):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device
        kwargs = {}
        if device:
            kwargs["device"] = device
        self._model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> List[list]:
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        ).tolist()
