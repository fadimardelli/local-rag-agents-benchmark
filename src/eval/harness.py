import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
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
    AGENTIC_MAX_ITERS,
    CONTRACTNLI_ORIG_INDEX_PATH,
    CONTRACTNLI_ORIG_META_PATH,
    HOTPOTQA_INDEX_PATH,
    HOTPOTQA_META_PATH,
)
from src.utils.memory import PeakMemory
from src.utils.timers import Timer

THESIS_SUMMARY_KEYS = [
    "count",
    "answer_accuracy",
    "answer_f1_mean",
    "evidence_recall_mean",
    "topk_hit_rate",
    "supporting_doc_recall_mean",
    "latency_median_s",
    "latency_p95_s",
    "latency_max_s",
    "peak_rss_bytes_max",
    "retrieval_calls_mean",
    "iterations_mean",
    "iterations_max",
    "max_iterations_hit_rate",
    "retrieved_context_tokens_mean",
    "prompt_tokens_mean",
    "answer_tokens_mean",
    "retrieval_latency_mean_s",
    "generation_latency_mean_s",
]

CALIBRATION_SUMMARY_KEYS = [
    "queries_in_batch",
    "success_count",
    "timeout_count",
    "error_count",
    "timeout_rate",
    "error_rate",
    "batch_start_time_utc",
    "batch_end_time_utc",
    "batch_first_k_latency_median_s",
    "batch_first_third_latency_median_s",
    "batch_last_third_latency_median_s",
    "batch_latency_drift_abs_s",
    "batch_latency_drift_pct",
]


@dataclass
class EvalResult:
    run_id: str
    batch_id: str
    query_index_in_batch: int
    timestamp_start: str
    timestamp_end: str
    query: str
    answer: str
    correct: Optional[bool]
    evidence_recall: bool
    topk_hit: bool
    latency_s: float
    peak_rss_bytes: int
    retrieval_calls: int
    retrieved_context_chars: int
    retrieved_context_tokens: int
    iterations: int
    prompt_chars: int
    prompt_tokens: int
    answer_chars: int
    answer_tokens: int
    retrieval_latency_s: float
    generation_latency_s: float
    timed_out: bool = False
    error: bool = False
    error_message: Optional[str] = None
    success_flag: bool = True
    timeout_flag: bool = False
    error_flag: bool = False
    supporting_doc_recall: Optional[float] = None
    answer_f1: Optional[float] = None
    iteration_trace: Optional[list] = None


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
    run_id: str = "",
    batch_id: str = "",
    trace_agentic: bool = False,
    top_k: int = 5,
) -> List[EvalResult]:
    if mode not in {"traditional", "agentic"}:
        raise ValueError("mode must be 'traditional' or 'agentic'")

    cases_list = list(cases)
    results: List[EvalResult] = []
    rag = (
        TraditionalRAG(model_path=model_path)
        if mode == "traditional"
        else AgenticRAG(model_path=model_path, trace_iterations=trace_agentic)
    )

    if warmup and cases_list:
        for case in cases_list[:2]:
            _ = rag.run(case.query, top_k=top_k)

    for idx, case in enumerate(cases_list):
        started_at = datetime.now(timezone.utc)
        with PeakMemory() as mem, Timer() as timer:
            if mode == "traditional":
                out = rag.run(case.query, top_k=top_k)
                iterations = 1
                retrieval_calls = 1
            else:
                out = rag.run(case.query, top_k=top_k)
                iterations = out.iterations
                retrieval_calls = out.iterations
        ended_at = datetime.now(timezone.utc)

        used = out.used
        retrieved_context_chars = _estimate_context_chars(used)
        prompt_chars = _estimate_prompt_chars(retrieved_context_chars, case.query)
        answer_chars = len(out.answer)
        correct = answer_correctness(out.answer, case.gold_answer)
        topk_hit = evidence_recall_multi_file(
            used_chunks=out.retrieved,
            gold_spans=case.gold_spans,
        )
        ev_recall = evidence_recall_multi_file(
            used_chunks=used,
            gold_spans=case.gold_spans,
        )

        results.append(
            EvalResult(
                run_id=run_id,
                batch_id=batch_id,
                query_index_in_batch=idx,
                timestamp_start=started_at.isoformat(),
                timestamp_end=ended_at.isoformat(),
                query=case.query,
                answer=out.answer,
                correct=correct,
                evidence_recall=ev_recall,
                topk_hit=topk_hit,
                latency_s=timer.elapsed,
                peak_rss_bytes=mem.peak_rss_bytes,
                retrieval_calls=retrieval_calls,
                retrieved_context_chars=retrieved_context_chars,
                retrieved_context_tokens=out.retrieved_context_tokens,
                iterations=iterations,
                prompt_chars=prompt_chars,
                prompt_tokens=out.prompt_tokens,
                answer_chars=answer_chars,
                answer_tokens=out.answer_tokens,
                retrieval_latency_s=out.retrieval_latency_s,
                generation_latency_s=out.generation_latency_s,
                supporting_doc_recall=None,
                answer_f1=None,
                iteration_trace=out.iteration_trace if mode == "agentic" and trace_agentic else None,
            )
        )

    return results


def run_eval_contractnli_original(
    cases: Iterable[ContractNLIOriginalCase],
    mode: str,
    label_mode: bool = False,
    warmup: bool = False,
    model_path: Optional[Path] = None,
    run_id: str = "",
    batch_id: str = "",
    top_k: int = 5,
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
        for case in cases_list[:2]:
            if mode == "traditional":
                _ = rag.run_label(case.query, top_k=top_k) if label_mode else rag.run(case.query, top_k=top_k)
            else:
                _ = rag.run_label(case.query, top_k=top_k) if label_mode else rag.run(case.query, top_k=top_k)

    results: List[EvalResult] = []
    for idx, case in enumerate(cases_list):
        started_at = datetime.now(timezone.utc)
        with PeakMemory() as mem, Timer() as timer:
            if mode == "traditional":
                out = rag.run_label(case.query, top_k=top_k) if label_mode else rag.run(case.query, top_k=top_k)
                iterations = 1
                retrieval_calls = 1
            else:
                out = rag.run_label(case.query, top_k=top_k) if label_mode else rag.run(case.query, top_k=top_k)
                iterations = out.iterations
                retrieval_calls = out.iterations
        ended_at = datetime.now(timezone.utc)

        used = out.used
        retrieved_context_chars = _estimate_context_chars(used)
        prompt_chars = _estimate_prompt_chars(retrieved_context_chars, case.query)
        answer_chars = len(out.answer)
        correct = answer_correctness(out.answer, case.gold_label)
        topk_hit = evidence_recall_multi(
            used_chunks=out.retrieved,
            gold_file_name=case.file_name,
            gold_spans=case.gold_spans,
        )
        ev_recall = evidence_recall_multi(
            used_chunks=used,
            gold_file_name=case.file_name,
            gold_spans=case.gold_spans,
        )

        results.append(
            EvalResult(
                run_id=run_id,
                batch_id=batch_id,
                query_index_in_batch=idx,
                timestamp_start=started_at.isoformat(),
                timestamp_end=ended_at.isoformat(),
                query=case.query,
                answer=out.answer,
                correct=correct,
                evidence_recall=ev_recall,
                topk_hit=topk_hit,
                latency_s=timer.elapsed,
                peak_rss_bytes=mem.peak_rss_bytes,
                retrieval_calls=retrieval_calls,
                retrieved_context_chars=retrieved_context_chars,
                retrieved_context_tokens=out.retrieved_context_tokens,
                iterations=iterations,
                prompt_chars=prompt_chars,
                prompt_tokens=out.prompt_tokens,
                answer_chars=answer_chars,
                answer_tokens=out.answer_tokens,
                retrieval_latency_s=out.retrieval_latency_s,
                generation_latency_s=out.generation_latency_s,
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
    run_id: str = "",
    batch_id: str = "",
    top_k: int = 5,
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
        for case in cases_list[:2]:
            _ = rag.run_hotpot(case.query, top_k=top_k)

    results: List[EvalResult] = []
    for idx, case in enumerate(cases_list):
        started_at = datetime.now(timezone.utc)
        with PeakMemory() as mem, Timer() as timer:
            if mode == "traditional":
                out = rag.run_hotpot(case.query, top_k=top_k)
                iterations = 1
                retrieval_calls = 1
            else:
                out = rag.run_hotpot(case.query, top_k=top_k)
                iterations = out.iterations
                retrieval_calls = out.iterations
        ended_at = datetime.now(timezone.utc)

        used = out.used
        retrieved_context_chars = _estimate_context_chars(used)
        prompt_chars = _estimate_prompt_chars(retrieved_context_chars, case.query)
        answer_chars = len(out.answer)
        correct = answer_exact_match(out.answer, case.gold_answer)
        f1 = answer_f1(out.answer, case.gold_answer)
        supp_recall = supporting_doc_recall(used, case.supporting_titles)
        topk_hit = supporting_doc_recall(out.retrieved, case.supporting_titles) > 0.0
        ev_recall = supp_recall == 1.0

        results.append(
            EvalResult(
                run_id=run_id,
                batch_id=batch_id,
                query_index_in_batch=idx,
                timestamp_start=started_at.isoformat(),
                timestamp_end=ended_at.isoformat(),
                query=case.query,
                answer=out.answer,
                correct=correct,
                evidence_recall=ev_recall,
                topk_hit=topk_hit,
                latency_s=timer.elapsed,
                peak_rss_bytes=mem.peak_rss_bytes,
                retrieval_calls=retrieval_calls,
                retrieved_context_chars=retrieved_context_chars,
                retrieved_context_tokens=out.retrieved_context_tokens,
                iterations=iterations,
                prompt_chars=prompt_chars,
                prompt_tokens=out.prompt_tokens,
                answer_chars=answer_chars,
                answer_tokens=out.answer_tokens,
                retrieval_latency_s=out.retrieval_latency_s,
                generation_latency_s=out.generation_latency_s,
                supporting_doc_recall=supp_recall,
                answer_f1=f1,
            )
        )

    return results


def summarize(results: List[EvalResult]) -> Dict[str, float]:
    if not results:
        return {}
    latencies = [r.latency_s for r in results]
    one_third = max(1, len(results) // 3)
    first_third_latencies = [r.latency_s for r in results[:one_third]]
    last_third_latencies = [r.latency_s for r in results[-one_third:]]
    evidence = [1.0 if r.evidence_recall else 0.0 for r in results]
    topk_hits = [1.0 if r.topk_hit else 0.0 for r in results]
    correct_vals = [r.correct for r in results if r.correct is not None]
    supporting_vals = [r.supporting_doc_recall for r in results if r.supporting_doc_recall is not None]
    f1_vals = [r.answer_f1 for r in results if r.answer_f1 is not None]
    timeout_vals = [1.0 if r.timed_out else 0.0 for r in results]
    error_vals = [1.0 if r.error else 0.0 for r in results]
    summary = {
        "count": float(len(results)),
        "queries_in_batch": float(len(results)),
        "success_count": float(sum(1.0 for r in results if r.success_flag)),
        "timeout_count": float(sum(1.0 for r in results if r.timeout_flag)),
        "error_count": float(sum(1.0 for r in results if r.error_flag)),
        "batch_start_time_utc": results[0].timestamp_start,
        "batch_end_time_utc": results[-1].timestamp_end,
        "latency_median_s": float(statistics.median(latencies)),
        "latency_p95_s": float(statistics.quantiles(latencies, n=20)[-1])
        if len(latencies) >= 20
        else float(max(latencies)),
        "latency_max_s": float(max(latencies)),
        "evidence_recall_mean": float(sum(evidence) / len(evidence)),
        "topk_hit_rate": float(sum(topk_hits) / len(topk_hits)),
        "timeout_rate": float(sum(timeout_vals) / len(timeout_vals)),
        "error_rate": float(sum(error_vals) / len(error_vals)),
        "peak_rss_bytes_max": float(max(r.peak_rss_bytes for r in results)),
        "retrieval_calls_mean": float(sum(r.retrieval_calls for r in results) / len(results)),
        "iterations_mean": float(sum(r.iterations for r in results) / len(results)),
        "iterations_max": float(max(r.iterations for r in results)),
        "max_iterations_hit_rate": float(
            sum(1.0 for r in results if r.iterations >= AGENTIC_MAX_ITERS) / len(results)
        ),
        "retrieved_context_chars_mean": float(
            sum(r.retrieved_context_chars for r in results) / len(results)
        ),
        "retrieved_context_tokens_mean": float(
            sum(r.retrieved_context_tokens for r in results) / len(results)
        ),
        "prompt_chars_mean": float(sum(r.prompt_chars for r in results) / len(results)),
        "prompt_tokens_mean": float(sum(r.prompt_tokens for r in results) / len(results)),
        "answer_chars_mean": float(sum(r.answer_chars for r in results) / len(results)),
        "answer_tokens_mean": float(sum(r.answer_tokens for r in results) / len(results)),
        "retrieval_latency_mean_s": float(sum(r.retrieval_latency_s for r in results) / len(results)),
        "generation_latency_mean_s": float(sum(r.generation_latency_s for r in results) / len(results)),
        "batch_first_k_latency_median_s": float(statistics.median(latencies[: min(5, len(latencies))])),
        "batch_first_third_latency_median_s": float(statistics.median(first_third_latencies)),
        "batch_last_third_latency_median_s": float(statistics.median(last_third_latencies)),
    }
    summary["batch_latency_drift_abs_s"] = (
        summary["batch_last_third_latency_median_s"] - summary["batch_first_third_latency_median_s"]
    )
    summary["batch_latency_drift_pct"] = (
        (summary["batch_latency_drift_abs_s"] / summary["batch_first_third_latency_median_s"])
        if summary["batch_first_third_latency_median_s"] > 0
        else 0.0
    )
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


def select_summary_view(summary: Dict[str, float], view: str) -> Dict[str, float]:
    if view == "all":
        return summary
    if view == "thesis":
        return {k: summary[k] for k in THESIS_SUMMARY_KEYS if k in summary}
    if view == "calibration":
        return {k: summary[k] for k in CALIBRATION_SUMMARY_KEYS if k in summary}
    raise ValueError("view must be one of: all, thesis, calibration")
