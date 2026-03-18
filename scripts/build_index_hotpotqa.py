import json
from pathlib import Path
from typing import Dict, List

from src.config.defaults import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
    HOTPOTQA_CORPUS_PATH,
    HOTPOTQA_INDEX_PATH,
    HOTPOTQA_META_PATH,
)
from src.data.chunking import chunk_text
from src.indexing import EmbeddingModel, build_faiss_index, save_faiss_index, save_metadata


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    docs = _read_jsonl(HOTPOTQA_CORPUS_PATH)

    metadata = []
    for doc in docs:
        title = doc.get("title")
        text = doc.get("text", "")
        if not title or not text:
            continue
        for ch in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            metadata.append(
                {
                    "doc_path": title,
                    "start": ch.start,
                    "end": ch.end,
                    "text": ch.text,
                }
            )

    embedder = EmbeddingModel(EMBEDDING_MODEL_NAME, normalize=EMBEDDING_NORMALIZE)
    vectors = embedder.encode([m["text"] for m in metadata])
    index = build_faiss_index(vectors, normalize=EMBEDDING_NORMALIZE)
    save_faiss_index(index, HOTPOTQA_INDEX_PATH)
    save_metadata(metadata, HOTPOTQA_META_PATH)

    print(f"Index saved to: {HOTPOTQA_INDEX_PATH}")
    print(f"Metadata saved to: {HOTPOTQA_META_PATH}")
    print(f"Chunks indexed: {len(metadata)}")


if __name__ == "__main__":
    main()
