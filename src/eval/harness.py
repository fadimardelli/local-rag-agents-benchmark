import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.eval.datasets import ContractNLITestCase, ContractNLIOriginalCase, HotpotQACase
from src.eval.metrics import (
    answer_correctness,
    answer_exact_match,
    answer_f1,
    evidence_recall_multi,
    evidence_recall_multi_file,
    supporting_doc_recall,
)

from src.rag import AgenticRAG, TraditionalRAG, Retriever
from src.config.defaults import (
    CONTRACTNLI_ORIG_INDEX_PATH,
    CONTRACTNLI_ORIG_META_PATH,
    HOTPOTQA_INDEX_PATH,
    HOTPOTQA_META_PATH,
)
from src.utils.memory import PeakMemory
from src.utils.timers import Timer


@dataclass
class EvalResult:
    query: str
    answer: str
    correct: Optional[bool]
    evidence_recall: bool
    latency_s: float
    peak_rss_bytes: int
    retrieval_calls: int
    retrieved_context_chars: int
    iterations: int
    prompt_chars: int
    answer_chars: int
    supporting_doc_recall: Optional[float] = None
    answer_f1: Optional[float] = None


def _estimate_context_chars(chunks: List) -> int:
    if not chunks:
        return 0
    total = sum(len(c.text) for c in chunks)
    separators = 2 * (len(chunks) - 1)
    return total + separators


def _estimate_prompt_chars(context_chars: int, question: str) -> int:
    return len("Context:\n") + context_chars + len("\n\nQuestion:\n") + len(question) + len("\n\nAnswer:")


def run_eval(
    cases: Iterable[ContractNLITestCase],
    mode: str,
    warmup: bool = False,
    model_path: Optional[Path] = None,
) -> List[EvalResult]:
    if mode not in {"traditional", "agentic"}:
        raise ValueError("mode must be 'traditional' or 'agentic'")

    cases_list = list(cases)
    results: List[EvalResult] = []
    rag = (
        TraditionalRAG(model_path=model_path)
        if mode == "traditional"
        else AgenticRAG(model_path=model_path)
    )

    if warmup and cases_list:
        _ = rag.run_hotpot(cases_list[0].query)

    for case in cases_list:
        with PeakMemory() as mem, Timer() as timer:
            if mode == "traditional":
                out = rag.run(case.query)
                iterations = 1
                retrieval_calls = 1
            else:
                out = rag.run(case.query)
                iterations = out.iterations
                retrieval_calls = out.iterations

        used = out.used
        retrieved_context_chars = _estimate_context_chars(used)
        prompt_chars = _estimate_prompt_chars(retrieved_context_chars, case.query)
        answer_chars = len(out.answer)
        correct = answer_correctness(out.answer, case.gold_answer)
        ev_recall = evidence_recall_multi_file(
            used_chunks=used,
            gold_spans=case.gold_spans,
        )

        results.append(
            EvalResult(
                query=case.query,
                answer=out.answer,
                correct=correct,
                evidence_recall=ev_recall,
                latency_s=timer.elapsed,
                peak_rss_bytes=mem.peak_rss_bytes,
                retrieval_calls=retrieval_calls,
                retrieved_context_chars=retrieved_context_chars,
                iterations=iterations,
                prompt_chars=prompt_chars,
                answer_chars=answer_chars,
                supporting_doc_recall=None,
                answer_f1=None,
            )
        )

    return results


def run_eval_contractnli_original(
    cases: Iterable[ContractNLIOriginalCase],
    mode: str,
    label_mode: bool = False,
    warmup: bool = False,
    model_path: Optional[Path] = None,
) -> List[EvalResult]:
    if mode not in {"traditional", "agentic"}:
        raise ValueError("mode must be 'traditional' or 'agentic'")

    cases_list = list(cases)
    retriever = Retriever(
        index_path=CONTRACTNLI_ORIG_INDEX_PATH,
        meta_path=CONTRACTNLI_ORIG_META_PATH,
    )
    rag = (
        TraditionalRAG(retriever=retriever, model_path=model_path)
        if mode == "traditional"
        else AgenticRAG(retriever=retriever, model_path=model_path)
    )

    if warmup and cases_list:
        first_case = cases_list[0]
        if first_case is not None:
            if mode == "traditional":
                _ = rag.run_label(first_case.query) if label_mode else rag.run(first_case.query)
            else:
                _ = rag.run_label(first_case.query) if label_mode else rag.run(first_case.query)

    results: List[EvalResult] = []
    for case in cases_list:
        with PeakMemory() as mem, Timer() as timer:
            if mode == "traditional":
                out = rag.run_label(case.query) if label_mode else rag.run(case.query)
                iterations = 1
                retrieval_calls = 1
            else:
                out = rag.run_label(case.query) if label_mode else rag.run(case.query)
                iterations = out.iterations
                retrieval_calls = out.iterations

        used = out.used
        retrieved_context_chars = _estimate_context_chars(used)
        prompt_chars = _estimate_prompt_chars(retrieved_context_chars, case.query)
        answer_chars = len(out.answer)
        correct = answer_correctness(out.answer, case.gold_label)
        ev_recall = evidence_recall_multi(
            used_chunks=used,
            gold_file_name=case.file_name,
            gold_spans=case.gold_spans,
        )

        results.append(
            EvalResult(
                query=case.query,
                answer=out.answer,
                correct=correct,
                evidence_recall=ev_recall,
                latency_s=timer.elapsed,
                peak_rss_bytes=mem.peak_rss_bytes,
                retrieval_calls=retrieval_calls,
                retrieved_context_chars=retrieved_context_chars,
                iterations=iterations,
                prompt_chars=prompt_chars,
                answer_chars=answer_chars,
                supporting_doc_recall=None,
                answer_f1=None,
            )
        )

    return results


def run_eval_hotpotqa(
    cases: Iterable[HotpotQACase],
    mode: str,
    warmup: bool = False,
    model_path: Optional[Path] = None,
) -> List[EvalResult]:
    if mode not in {"traditional", "agentic"}:
        raise ValueError("mode must be 'traditional' or 'agentic'")

    cases_list = list(cases)
    retriever = Retriever(
        index_path=HOTPOTQA_INDEX_PATH,
        meta_path=HOTPOTQA_META_PATH,
    )
    rag = (
        TraditionalRAG(retriever=retriever, model_path=model_path)
        if mode == "traditional"
        else AgenticRAG(retriever=retriever, model_path=model_path)
    )

    if warmup and cases_list:
        _ = rag.run(cases_list[0].query)

    results: List[EvalResult] = []
    for case in cases_list:
        with PeakMemory() as mem, Timer() as timer:
            if mode == "traditional":
                out = rag.run_hotpot(case.query)
                iterations = 1
                retrieval_calls = 1
            else:
                out = rag.run_hotpot(case.query)
                iterations = out.iterations
                retrieval_calls = out.iterations

        used = out.used
        retrieved_context_chars = _estimate_context_chars(used)
        prompt_chars = _estimate_prompt_chars(retrieved_context_chars, case.query)
        answer_chars = len(out.answer)
        correct = answer_exact_match(out.answer, case.gold_answer)
        f1 = answer_f1(out.answer, case.gold_answer)
        supp_recall = supporting_doc_recall(used, case.supporting_titles)
        ev_recall = supp_recall == 1.0

        results.append(
            EvalResult(
                query=case.query,
                answer=out.answer,
                correct=correct,
                evidence_recall=ev_recall,
                latency_s=timer.elapsed,
                peak_rss_bytes=mem.peak_rss_bytes,
                retrieval_calls=retrieval_calls,
                retrieved_context_chars=retrieved_context_chars,
                iterations=iterations,
                prompt_chars=prompt_chars,
                answer_chars=answer_chars,
                supporting_doc_recall=supp_recall,
                answer_f1=f1,
            )
        )

    return results


def summarize(results: List[EvalResult]) -> Dict[str, float]:
    if not results:
        return {}
    latencies = [r.latency_s for r in results]
    evidence = [1.0 if r.evidence_recall else 0.0 for r in results]
    correct_vals = [r.correct for r in results if r.correct is not None]
    supporting_vals = [r.supporting_doc_recall for r in results if r.supporting_doc_recall is not None]
    f1_vals = [r.answer_f1 for r in results if r.answer_f1 is not None]
    summary = {
        "count": float(len(results)),
        "latency_median_s": float(statistics.median(latencies)),
        "latency_p95_s": float(statistics.quantiles(latencies, n=20)[-1])
        if len(latencies) >= 20
        else float(max(latencies)),
        "evidence_recall_mean": float(sum(evidence) / len(evidence)),
        "peak_rss_bytes_max": float(max(r.peak_rss_bytes for r in results)),
        "retrieval_calls_mean": float(sum(r.retrieval_calls for r in results) / len(results)),
        "iterations_mean": float(sum(r.iterations for r in results) / len(results)),
        "retrieved_context_chars_mean": float(
            sum(r.retrieved_context_chars for r in results) / len(results)
        ),
        "prompt_chars_mean": float(sum(r.prompt_chars for r in results) / len(results)),
        "answer_chars_mean": float(sum(r.answer_chars for r in results) / len(results)),
    }
    if correct_vals:
        summary["answer_accuracy"] = float(sum(1 for c in correct_vals if c) / len(correct_vals))
    if supporting_vals:
        summary["supporting_doc_recall_mean"] = float(sum(supporting_vals) / len(supporting_vals))
    if f1_vals:
        summary["answer_f1_mean"] = float(sum(f1_vals) / len(f1_vals))
    return summary


def write_results(results: List[EvalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


def write_results_csv(results: List[EvalResult], output_path: Path) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].__dict__.keys()) if results else []
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)


def write_summary_csv(summary: Dict[str, float], output_path: Path) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])
