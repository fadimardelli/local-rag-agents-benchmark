import re
from typing import List, Optional

from src.rag.retriever import RetrievedChunk


def normalize_label(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = text.strip().lower()
    t = re.sub(r"[^a-z]+", "", t)
    return t or None


def answer_correctness(pred: str, gold: Optional[str]) -> Optional[bool]:
    gold_norm = normalize_label(gold)
    if gold_norm is None:
        return None
    pred_norm = normalize_label(pred)
    if pred_norm is None:
        return False
    return pred_norm == gold_norm


def _doc_matches(retrieved_doc_path: str, gold_file_path: str) -> bool:
    # gold paths are like "contractnli/FILE.txt"
    return retrieved_doc_path.replace("\\", "/").endswith(gold_file_path)


def evidence_recall(
    used_chunks: List[RetrievedChunk],
    gold_file_path: str,
    gold_start: int,
    gold_end: int,
) -> bool:
    for chunk in used_chunks:
        if not _doc_matches(chunk.doc_path, gold_file_path):
            continue
        overlap = min(chunk.end, gold_end) - max(chunk.start, gold_start)
        if overlap > 0:
            return True
    return False


def evidence_recall_multi_file(
    used_chunks: List[RetrievedChunk],
    gold_spans: List[tuple],
) -> bool:
    if not gold_spans:
        return False
    for gold_file_path, gold_start, gold_end in gold_spans:
        for chunk in used_chunks:
            if not _doc_matches(chunk.doc_path, gold_file_path):
                continue
            overlap = min(chunk.end, gold_end) - max(chunk.start, gold_start)
            if overlap > 0:
                return True
    return False


def evidence_recall_multi(
    used_chunks: List[RetrievedChunk],
    gold_file_name: str,
    gold_spans: List[tuple],
) -> bool:
    if not gold_spans:
        return False
    for chunk in used_chunks:
        if not _doc_matches(chunk.doc_path, gold_file_name):
            continue
        for start, end in gold_spans:
            overlap = min(chunk.end, end) - max(chunk.start, start)
            if overlap > 0:
                return True
    return False
