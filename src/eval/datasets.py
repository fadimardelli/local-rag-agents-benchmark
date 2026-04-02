from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import json
from pathlib import Path

from src.data import (
    LegalBenchRAGData,
    load_legalbench_contractnli,
    load_legalbench_mini,
    load_legalbench_mini_balanced,
)
from src.config.defaults import CONTRACTNLI_ORIG_TEST_PATH, HOTPOTQA_CASES_PATH


@dataclass(frozen=True)
class ContractNLITestCase:
    case_id: str
    query: str
    gold_spans: List[Tuple[str, int, int]]
    gold_answer: Optional[str]


@dataclass(frozen=True)
class ContractNLIOriginalCase:
    query: str
    file_name: str
    gold_label: str
    gold_spans: List[Tuple[int, int]]


@dataclass(frozen=True)
class HotpotQACase:
    query: str
    gold_answer: str
    supporting_titles: List[str]


def _iter_legalbench_cases(
    data: LegalBenchRAGData,
    limit: Optional[int] = None,
    case_prefix: str = "legalbench_test",
) -> Iterable[ContractNLITestCase]:
    cases = data.test_cases
    if limit is not None:
        cases = cases[:limit]

    for idx, case in enumerate(cases):
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
            case_id=f"{case_prefix}_{idx:04d}",
            query=query,
            gold_spans=gold_spans,
            gold_answer=case.get("label"),
        )


def iter_contractnli_cases(limit: Optional[int] = None) -> Iterable[ContractNLITestCase]:
    data: LegalBenchRAGData = load_legalbench_contractnli()
    yield from _iter_legalbench_cases(data, limit=limit, case_prefix="contractnli_test")


def iter_legalbench_mini_cases(limit: Optional[int] = None) -> Iterable[ContractNLITestCase]:
    data: LegalBenchRAGData = load_legalbench_mini()
    yield from _iter_legalbench_cases(data, limit=limit, case_prefix="legalbench_mini_test")


def iter_legalbench_mini_balanced_cases(limit: Optional[int] = None) -> Iterable[ContractNLITestCase]:
    data: LegalBenchRAGData = load_legalbench_mini_balanced()
    yield from _iter_legalbench_cases(data, limit=limit, case_prefix="legalbench_mini_balanced_test")


def iter_contractnli_cases_window(
    offset: int = 0,
    limit: Optional[int] = None,
) -> Iterable[ContractNLITestCase]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    skipped = 0
    emitted = 0
    for case in iter_contractnli_cases(limit=None):
        if skipped < offset:
            skipped += 1
            continue
        yield case
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def iter_legalbench_mini_cases_window(
    offset: int = 0,
    limit: Optional[int] = None,
) -> Iterable[ContractNLITestCase]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    skipped = 0
    emitted = 0
    for case in iter_legalbench_mini_cases(limit=None):
        if skipped < offset:
            skipped += 1
            continue
        yield case
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def iter_legalbench_mini_balanced_cases_window(
    offset: int = 0,
    limit: Optional[int] = None,
) -> Iterable[ContractNLITestCase]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    skipped = 0
    emitted = 0
    for case in iter_legalbench_mini_balanced_cases(limit=None):
        if skipped < offset:
            skipped += 1
            continue
        yield case
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def iter_contractnli_original_cases(
    split_path: Path = CONTRACTNLI_ORIG_TEST_PATH,
    offset: int = 0,
    limit: Optional[int] = None,
) -> Iterable[ContractNLIOriginalCase]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing ContractNLI original split: {split_path}")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    documents = data.get("documents", [])
    labels = data.get("labels", {})

    skipped = 0
    emitted = 0
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
            if skipped < offset:
                skipped += 1
                continue
            yield ContractNLIOriginalCase(
                query=hypo,
                file_name=file_name or str(doc_id),
                gold_label=choice,
                gold_spans=gold_spans,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_hotpotqa_cases(
    cases_path: Path = HOTPOTQA_CASES_PATH,
    offset: int = 0,
    limit: Optional[int] = None,
) -> Iterable[HotpotQACase]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if not cases_path.exists():
        raise FileNotFoundError(f"Missing HotpotQA cases file: {cases_path}")

    count = 0
    skipped = 0
    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query = row.get("question")
            answer = row.get("answer")
            supporting_titles = row.get("supporting_titles") or []
            if not query or answer is None:
                continue
            if skipped < offset:
                skipped += 1
                continue
            yield HotpotQACase(
                query=query,
                gold_answer=answer,
                supporting_titles=supporting_titles,
            )
            count += 1
            if limit is not None and count >= limit:
                return
