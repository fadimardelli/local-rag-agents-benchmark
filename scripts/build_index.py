import argparse
from pathlib import Path

from src.config.defaults import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTRACTNLI_INDEX_PATH,
    CONTRACTNLI_META_PATH,
    CORPUS_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
)
from src.data import chunk_document, iter_corpus_files
from src.indexing import EmbeddingModel, build_faiss_index, save_faiss_index, save_metadata


def build_contractnli_index(device: str | None = None) -> None:
    embedder = EmbeddingModel(
        EMBEDDING_MODEL_NAME,
        normalize=EMBEDDING_NORMALIZE,
        device=device,
    )

    all_chunks = []
    for doc_path in iter_corpus_files(CORPUS_DIR):
        chunks = chunk_document(doc_path, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    texts = [c.text for c in all_chunks]
    vectors = embedder.encode(texts)

    index = build_faiss_index(vectors, normalize=EMBEDDING_NORMALIZE)
    save_faiss_index(index, CONTRACTNLI_INDEX_PATH)

    metadata = [
        {
            "doc_path": c.doc_path,
            "start": c.start,
            "end": c.end,
            "text": c.text,
        }
        for c in all_chunks
    ]
    save_metadata(metadata, CONTRACTNLI_META_PATH)

    print(f"Index saved to: {CONTRACTNLI_INDEX_PATH}")
    print(f"Metadata saved to: {CONTRACTNLI_META_PATH}")
    print(f"Chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    build_contractnli_index(device=args.device)
