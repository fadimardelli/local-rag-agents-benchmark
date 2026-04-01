import argparse
import json
from pathlib import Path

from src.config.defaults import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTRACTNLI_ORIG_INDEX_PATH,
    CONTRACTNLI_ORIG_META_PATH,
    CONTRACTNLI_ORIG_TEST_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
)
from src.data.chunking import chunk_text
from src.indexing import EmbeddingModel, build_faiss_index, save_faiss_index, save_metadata


def build_index_from_original(test_path: Path, device: str | None = None) -> None:
    if not test_path.exists():
        raise FileNotFoundError(f"Missing ContractNLI original split: {test_path}")

    data = json.loads(test_path.read_text(encoding="utf-8"))
    documents = data.get("documents", [])

    all_chunks = []
    for doc in documents:
        doc_id = doc.get("id")
        file_name = doc.get("file_name")
        text = doc.get("text", "")
        for ch in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append(
                {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "start": ch.start,
                    "end": ch.end,
                    "text": ch.text,
                }
            )

    embedder = EmbeddingModel(
        EMBEDDING_MODEL_NAME,
        normalize=EMBEDDING_NORMALIZE,
        device=device,
    )
    vectors = embedder.encode([c["text"] for c in all_chunks])
    index = build_faiss_index(vectors, normalize=EMBEDDING_NORMALIZE)
    save_faiss_index(index, CONTRACTNLI_ORIG_INDEX_PATH)
    save_metadata(all_chunks, CONTRACTNLI_ORIG_META_PATH)

    print(f"Index saved to: {CONTRACTNLI_ORIG_INDEX_PATH}")
    print(f"Metadata saved to: {CONTRACTNLI_ORIG_META_PATH}")
    print(f"Chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    build_index_from_original(CONTRACTNLI_ORIG_TEST_PATH, device=args.device)
