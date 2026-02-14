from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import json
from pathlib import Path

from src.data import LegalBenchRAGData, load_legalbench_contractnli
from src.config.defaults import CONTRACTNLI_ORIG_TEST_PATH


@dataclass(frozen=True)
class ContractNLITestCase:
    query: str
    gold_spans: List[Tuple[str, int, int]]
    gold_answer: Optional[str]


@dataclass(frozen=True)
class ContractNLIOriginalCase:
    query: str
    file_name: str
    gold_label: str
    gold_spans: List[Tuple[int, int]]


def iter_contractnli_cases(limit: Optional[int] = None) -> Iterable[ContractNLITestCase]:
    data: LegalBenchRAGData = load_legalbench_contractnli()
    cases = data.test_cases
    if limit is not None:
        cases = cases[:limit]

    for case in cases:
        query = case.get("query")
        snippets = case.get("snippets") or []
        if not query or not snippets:
            continue
        gold_spans: List[Tuple[str, int, int]] = []
        for snip in snippets:
            file_path = snip.get("file_path")
            span = snip.get("span")
            if file_path is None or not isinstance(span, list) or len(span) != 2:
                continue
            gold_spans.append((file_path, int(span[0]), int(span[1])))
        if not gold_spans:
            continue
        yield ContractNLITestCase(
            query=query,
            gold_spans=gold_spans,
            gold_answer=case.get("label"),
        )


def iter_contractnli_original_cases(
    split_path: Path = CONTRACTNLI_ORIG_TEST_PATH,
    limit: Optional[int] = None,
) -> Iterable[ContractNLIOriginalCase]:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing ContractNLI original split: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    documents = data.get("documents", [])
    labels = data.get("labels", {})

    # Build map of doc_id -> doc for faster access
    for doc in documents:
        doc_id = doc.get("id")
        file_name = doc.get("file_name")
        spans = doc.get("spans", [])
        ann_sets = doc.get("annotation_sets", [])
        if not ann_sets:
            continue
        annotations = ann_sets[0].get("annotations", {})

        for label_id, label_info in labels.items():
            hypo = label_info.get("hypothesis")
            ann = annotations.get(label_id)
            if hypo is None or ann is None:
                continue
            choice = ann.get("choice")
            span_idxs = ann.get("spans", [])
            gold_spans = []
            for idx in span_idxs:
                if 0 <= idx < len(spans):
                    start, end = spans[idx]
                    gold_spans.append((int(start), int(end)))
            yield ContractNLIOriginalCase(
                query=hypo,
                file_name=file_name or str(doc_id),
                gold_label=choice,
                gold_spans=gold_spans,
            )

            if limit is not None:
                limit -= 1
                if limit <= 0:
                    return
