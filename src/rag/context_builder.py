from typing import List, Tuple

from src.rag.retriever import RetrievedChunk


def build_context(
    chunks: List[RetrievedChunk],
    char_budget: int,
) -> Tuple[str, List[RetrievedChunk]]:
    selected = []
    total = 0
    parts = []

    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        sep = "\n\n"
        add_len = len(text) + (len(sep) if parts else 0)
        if total + add_len > char_budget:
            break
        if parts:
            parts.append(sep)
        parts.append(text)
        total += add_len
        selected.append(chunk)

    return "".join(parts), selected
