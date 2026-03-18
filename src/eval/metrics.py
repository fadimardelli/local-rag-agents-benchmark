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


def normalize_answer_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def answer_exact_match(pred: str, gold: str) -> bool:
    return normalize_answer_text(pred) == normalize_answer_text(gold)


def answer_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer_text(pred).split()
    gold_tokens = normalize_answer_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    common = 0
    for t, c in pred_counts.items():
        common += min(c, gold_counts.get(t, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


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


def supporting_doc_recall(used_chunks: List[RetrievedChunk], supporting_titles: List[str]) -> float:
    if not supporting_titles:
        return 0.0
    retrieved_titles = {c.doc_path for c in used_chunks}
    hit = sum(1 for t in supporting_titles if t in retrieved_titles)
    return hit / len(supporting_titles)
