from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
THESIS_ROOT = PROJECT_ROOT / "thesis"
OUT_ROOT = THESIS_ROOT / "reporting_pack"
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"


@dataclass(frozen=True)
class RunSpec:
    key: str
    benchmark: str
    mode: str
    model: str
    scope: str
    status: str
    label: str
    results_glob: str
    main_quality_metric: str
    main_quality_value_key: str
    notes: str = ""


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    frac = rank - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def load_jsonl_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(PROJECT_ROOT.glob(pattern)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(normalize_trace_costs(json.loads(line)))
    return rows


def normalize_trace_costs(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    trace = normalized.get("iteration_trace") or []
    if not trace:
        return normalized
    # Agentic retrieval_calls was historically recorded as iterations. When a
    # corrective backup query is used, the true retrieval count is one higher.
    # Guard on retrieval_calls == iterations so future reruns with fixed logging
    # are not corrected twice.
    backup_used = any("||" in (step.get("retrieval_query") or "") for step in trace)
    retrieval_calls = normalized.get("retrieval_calls")
    iterations = normalized.get("iterations")
    if (
        backup_used
        and isinstance(retrieval_calls, (int, float))
        and not isinstance(retrieval_calls, bool)
        and isinstance(iterations, (int, float))
        and not isinstance(iterations, bool)
        and int(retrieval_calls) == int(iterations)
    ):
        normalized["retrieval_calls"] = int(normalized["retrieval_calls"]) + 1
    return normalized


def mean_bool(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [1.0 if bool(r.get(key)) else 0.0 for r in rows if key in r and r.get(key) is not None]
    return mean(vals) if vals else None


def mean_num(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return mean(vals) if vals else None


def median_num(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return median(vals) if vals else None


def p95_num(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return percentile(vals, 0.95) if vals else None


def max_num(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return max(vals) if vals else None


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers = [str(r.get("answer", "")) for r in rows]
    return {
        "queries": len(rows),
        "answer_accuracy": mean_bool(rows, "correct"),
        "answer_f1": mean_num(rows, "answer_f1"),
        "evidence_recall": mean_bool(rows, "evidence_recall"),
        "topk_hit_rate": mean_bool(rows, "topk_hit"),
        "supporting_doc_recall": mean_num(rows, "supporting_doc_recall"),
        "latency_median_s": median_num(rows, "latency_s"),
        "latency_p95_s": p95_num(rows, "latency_s"),
        "latency_max_s": max_num(rows, "latency_s"),
        "peak_rss_bytes_max": max_num(rows, "peak_rss_bytes"),
        "retrieval_calls_mean": mean_num(rows, "retrieval_calls"),
        "iterations_mean": mean_num(rows, "iterations"),
        "iterations_max": max_num(rows, "iterations"),
        "retrieved_context_tokens_mean": mean_num(rows, "retrieved_context_tokens"),
        "prompt_tokens_mean": mean_num(rows, "prompt_tokens"),
        "answer_tokens_mean": mean_num(rows, "answer_tokens"),
        "retrieval_latency_mean_s": mean_num(rows, "retrieval_latency_s"),
        "generation_latency_mean_s": mean_num(rows, "generation_latency_s"),
        "unknown_like_count": sum(1 for a in answers if "i don't know" in a.lower()),
        "verbose_gt_12_words_count": sum(1 for a in answers if len(a.split()) > 12),
        "multiline_count": sum(1 for a in answers if "\n" in a),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, digits)
    return value


def bytes_to_gb(value: float | None) -> float | None:
    if value is None:
        return None
    return value / (1024 ** 3)


def load_legal_subset_map() -> dict[str, str]:
    benchmark_path = PROJECT_ROOT / "data" / "benchmarks" / "legalbenchrag_mini.json"
    obj = json.loads(benchmark_path.read_text(encoding="utf-8"))
    tests = obj.get("tests") or obj.get("test_cases") or []
    query_to_subset: dict[str, str] = {}
    for case in tests:
        query = case.get("query")
        tags = case.get("tags") or []
        subset = tags[0] if tags else "unknown"
        if query:
            query_to_subset[query] = subset
    return query_to_subset


def build_system_configurations(run_specs: list[RunSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in run_specs:
        if spec.benchmark == "LegalBench-mini":
            corrective = "none" if spec.mode == "traditional" else "doc-narrow corrective retrieval"
        elif spec.benchmark == "HotpotQA":
            corrective = "none" if spec.mode == "traditional" else "global corrective retrieval + answer selection"
        else:
            corrective = "unknown"
        rows.append(
            {
                "system_key": spec.key,
                "label": spec.label,
                "benchmark": spec.benchmark,
                "mode": spec.mode,
                "model": spec.model,
                "retriever": "shared fixed hybrid/dense local retriever",
                "corrective_step": corrective,
                "scope": spec.scope,
                "status": spec.status,
                "notes": spec.notes,
            }
        )
    return rows


def main() -> None:
    run_specs = [
        RunSpec(
            key="legal_trad_llama_3b",
            benchmark="LegalBench-mini",
            mode="traditional",
            model="Llama 3B",
            scope="full_776",
            status="ready",
            label="Traditional Llama 3B",
            results_glob="data/results/legalbench_mini/final/overnight_legalbench_mini_traditional_3b_off*.jsonl",
            main_quality_metric="evidence_recall",
            main_quality_value_key="evidence_recall",
        ),
        RunSpec(
            key="legal_agentic_llama_3b",
            benchmark="LegalBench-mini",
            mode="agentic",
            model="Llama 3B",
            scope="full_776",
            status="ready_extra",
            label="Agentic Llama 3B",
            results_glob="data/results/legalbench_mini/final/legalbench_mini_agentic_llama_3b_off*.jsonl",
            main_quality_metric="evidence_recall",
            main_quality_value_key="evidence_recall",
            notes="Full extra model-choice run.",
        ),
        RunSpec(
            key="legal_trad_llama_8b",
            benchmark="LegalBench-mini",
            mode="traditional",
            model="Llama 8B",
            scope="full_776",
            status="ready",
            label="Traditional Llama 8B",
            results_glob="data/results/legalbench_mini/final/overnight_legalbench_mini_traditional_8b_off*.jsonl",
            main_quality_metric="evidence_recall",
            main_quality_value_key="evidence_recall",
        ),
        RunSpec(
            key="legal_agentic_llama_8b",
            benchmark="LegalBench-mini",
            mode="agentic",
            model="Llama 8B",
            scope="full_776",
            status="ready",
            label="Agentic Llama 8B",
            results_glob="data/results/legalbench_mini/reruns/agentic_llama_8b_tracefix/legalbench_mini_agentic_llama_8b_tracefix_off*.jsonl",
            main_quality_metric="evidence_recall",
            main_quality_value_key="evidence_recall",
        ),
        RunSpec(
            key="legal_trad_qwen_3b",
            benchmark="LegalBench-mini",
            mode="traditional",
            model="Qwen 2.5 3B",
            scope="full_776",
            status="ready_extra",
            label="Traditional Qwen 3B",
            results_glob="data/results/legalbench_mini/final/legalbench_mini_traditional_qwen25_3b_off*.jsonl",
            main_quality_metric="evidence_recall",
            main_quality_value_key="evidence_recall",
            notes="Full extra model-choice run.",
        ),
        RunSpec(
            key="legal_trad_qwen_7b",
            benchmark="LegalBench-mini",
            mode="traditional",
            model="Qwen 2.5 7B",
            scope="full_776",
            status="ready_extra",
            label="Traditional Qwen 7B",
            results_glob="data/results/legalbench_mini/final/legalbench_mini_traditional_qwen25_7b_off*.jsonl",
            main_quality_metric="evidence_recall",
            main_quality_value_key="evidence_recall",
            notes="Full extra model-choice run.",
        ),
        RunSpec(
            key="legal_agentic_qwen_7b",
            benchmark="LegalBench-mini",
            mode="agentic",
            model="Qwen 2.5 7B",
            scope="full_776",
            status="ready_extra",
            label="Agentic Qwen 7B",
            results_glob="data/results/legalbench_mini/final/legalbench_mini_agentic_qwen25_7b_off*.jsonl",
            main_quality_metric="evidence_recall",
            main_quality_value_key="evidence_recall",
            notes="Full extra model-choice run.",
        ),
        RunSpec(
            key="legal_agentic_qwen_3b",
            benchmark="LegalBench-mini",
            mode="agentic",
            model="Qwen 2.5 3B",
            scope="full_776",
            status="ready_extra",
            label="Agentic Qwen 3B",
            results_glob="data/results/legalbench_mini/final/legalbench_mini_agentic_qwen25_3b_off*.jsonl",
            main_quality_metric="evidence_recall",
            main_quality_value_key="evidence_recall",
            notes="Full extra model-choice run.",
        ),
        RunSpec(
            key="hotpot_trad_llama_3b",
            benchmark="HotpotQA",
            mode="traditional",
            model="Llama 3B",
            scope="full_1000",
            status="ready",
            label="Traditional Llama 3B",
            results_glob="data/results/hotpotqa/reruns/traditional_llama_3b_shortanswerfix/hotpotqa_first1000_traditional_llama_3b_shortanswerfix_off*.jsonl",
            main_quality_metric="answer_accuracy",
            main_quality_value_key="answer_accuracy",
        ),
        RunSpec(
            key="hotpot_trad_llama_8b",
            benchmark="HotpotQA",
            mode="traditional",
            model="Llama 8B",
            scope="full_1000",
            status="ready",
            label="Traditional Llama 8B",
            results_glob="data/results/hotpotqa/reruns/traditional_llama_8b_shortanswerfix/hotpotqa_first1000_traditional_llama_8b_shortanswerfix_off*.jsonl",
            main_quality_metric="answer_accuracy",
            main_quality_value_key="answer_accuracy",
        ),
        RunSpec(
            key="hotpot_agentic_llama_3b",
            benchmark="HotpotQA",
            mode="agentic",
            model="Llama 3B",
            scope="full_1000",
            status="ready_extra",
            label="Agentic Llama 3B",
            results_glob="data/results/hotpotqa/reruns/agentic_llama_3b_selectorfix/hotpotqa_first1000_agentic_llama_3b_selectorfix_off*.jsonl",
            main_quality_metric="answer_accuracy",
            main_quality_value_key="answer_accuracy",
            notes="Full extra model-choice run under corrected selector and retrieval-count setup.",
        ),
        RunSpec(
            key="hotpot_trad_qwen_3b",
            benchmark="HotpotQA",
            mode="traditional",
            model="Qwen 2.5 3B",
            scope="full_1000",
            status="ready_extra",
            label="Traditional Qwen 3B",
            results_glob="data/results/hotpotqa/final/hotpotqa_first1000_traditional_qwen25_3b_off*.jsonl",
            main_quality_metric="answer_accuracy",
            main_quality_value_key="answer_accuracy",
            notes="Full extra model-choice run.",
        ),
        RunSpec(
            key="hotpot_trad_qwen_7b",
            benchmark="HotpotQA",
            mode="traditional",
            model="Qwen 2.5 7B",
            scope="full_1000",
            status="ready_extra",
            label="Traditional Qwen 7B",
            results_glob="data/results/hotpotqa/final/hotpotqa_first1000_traditional_qwen25_7b_off*.jsonl",
            main_quality_metric="answer_accuracy",
            main_quality_value_key="answer_accuracy",
            notes="Full extra model-choice run.",
        ),
        RunSpec(
            key="hotpot_agentic_qwen_3b",
            benchmark="HotpotQA",
            mode="agentic",
            model="Qwen 2.5 3B",
            scope="full_1000",
            status="ready_extra",
            label="Agentic Qwen 3B",
            results_glob="data/results/hotpotqa/reruns/agentic_qwen_3b_selectorfix/hotpotqa_first1000_agentic_qwen25_3b_selectorfix_off*.jsonl",
            main_quality_metric="answer_accuracy",
            main_quality_value_key="answer_accuracy",
            notes="Full extra model-choice rerun with corrected selector and retrieval-count setup.",
        ),
        RunSpec(
            key="hotpot_agentic_llama_8b",
            benchmark="HotpotQA",
            mode="agentic",
            model="Llama 8B",
            scope="full_1000",
            status="ready",
            label="Agentic Llama 8B",
            results_glob="data/results/hotpotqa/final/hotpotqa_first1000_agentic_8b_shortprompt_off*.jsonl",
            main_quality_metric="answer_accuracy",
            main_quality_value_key="answer_accuracy",
        ),
        RunSpec(
            key="hotpot_agentic_qwen_7b",
            benchmark="HotpotQA",
            mode="agentic",
            model="Qwen 2.5 7B",
            scope="full_1000",
            status="ready_extra",
            label="Agentic Qwen 7B",
            results_glob="data/results/hotpotqa/reruns/agentic_qwen_7b_selectorfix/hotpotqa_first1000_agentic_qwen25_7b_selectorfix_off*.jsonl",
            main_quality_metric="answer_accuracy",
            main_quality_value_key="answer_accuracy",
            notes="Full extra model-choice rerun with corrected selector and retrieval-count setup.",
        ),
    ]

    run_rows: dict[str, list[dict[str, Any]]] = {}
    run_summary: dict[str, dict[str, Any]] = {}
    for spec in run_specs:
        rows = load_jsonl_rows(spec.results_glob)
        run_rows[spec.key] = rows
        if rows:
            run_summary[spec.key] = summarize_run(rows)
        else:
            run_summary[spec.key] = {}

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # System configuration and run registry
    system_rows = build_system_configurations(run_specs)
    write_csv(
        TABLES_DIR / "table_01_system_configurations.csv",
        system_rows,
        ["system_key", "label", "benchmark", "mode", "model", "retriever", "corrective_step", "scope", "status", "notes"],
    )

    registry_rows = []
    for spec in run_specs:
        rows = run_rows[spec.key]
        registry_rows.append(
            {
                "system_key": spec.key,
                "label": spec.label,
                "benchmark": spec.benchmark,
                "mode": spec.mode,
                "model": spec.model,
                "scope": spec.scope,
                "status": spec.status,
                "files_found": len(list(PROJECT_ROOT.glob(spec.results_glob))),
                "rows_loaded": len(rows),
                "results_glob": spec.results_glob,
                "notes": spec.notes,
            }
        )
    write_csv(
        TABLES_DIR / "run_registry.csv",
        registry_rows,
        ["system_key", "label", "benchmark", "mode", "model", "scope", "status", "files_found", "rows_loaded", "results_glob", "notes"],
    )

    # Main legal table
    legal_keys = [
        "legal_trad_llama_3b",
        "legal_agentic_llama_3b",
        "legal_trad_llama_8b",
        "legal_agentic_llama_8b",
        "legal_trad_qwen_3b",
        "legal_trad_qwen_7b",
        "legal_agentic_qwen_3b",
        "legal_agentic_qwen_7b",
    ]
    legal_main_rows = []
    for key in legal_keys:
        spec = next(s for s in run_specs if s.key == key)
        s = run_summary[key]
        legal_main_rows.append(
            {
                "system": spec.label,
                "model": spec.model,
                "mode": spec.mode,
                "scope": spec.scope,
                "status": spec.status,
                "queries": s.get("queries", 0),
                "evidence_recall": fmt(s.get("evidence_recall")),
                "topk_hit_rate": fmt(s.get("topk_hit_rate")),
                "latency_median_s": fmt(s.get("latency_median_s"), 2),
                "latency_p95_s": fmt(s.get("latency_p95_s"), 2),
                "peak_memory_gb_max": fmt(bytes_to_gb(s.get("peak_rss_bytes_max")), 2),
                "retrieval_calls_mean": fmt(s.get("retrieval_calls_mean"), 3),
                "generation_latency_mean_s": fmt(s.get("generation_latency_mean_s"), 2),
                "notes": spec.notes,
            }
        )
    write_csv(
        TABLES_DIR / "table_03_legal_main_results.csv",
        legal_main_rows,
        ["system", "model", "mode", "scope", "status", "queries", "evidence_recall", "topk_hit_rate", "latency_median_s", "latency_p95_s", "peak_memory_gb_max", "retrieval_calls_mean", "generation_latency_mean_s", "notes"],
    )

    # Main hotpot table
    hotpot_keys = [
        "hotpot_trad_llama_3b",
        "hotpot_trad_llama_8b",
        "hotpot_agentic_llama_3b",
        "hotpot_trad_qwen_3b",
        "hotpot_trad_qwen_7b",
        "hotpot_agentic_qwen_3b",
        "hotpot_agentic_llama_8b",
        "hotpot_agentic_qwen_7b",
    ]
    hotpot_main_rows = []
    for key in hotpot_keys:
        spec = next(s for s in run_specs if s.key == key)
        s = run_summary[key]
        hotpot_main_rows.append(
            {
                "system": spec.label,
                "model": spec.model,
                "mode": spec.mode,
                "scope": spec.scope,
                "status": spec.status,
                "queries": s.get("queries", 0),
                "answer_accuracy": fmt(s.get("answer_accuracy")),
                "answer_f1": fmt(s.get("answer_f1")),
                "evidence_recall": fmt(s.get("evidence_recall")),
                "supporting_doc_recall": fmt(s.get("supporting_doc_recall")),
                "latency_median_s": fmt(s.get("latency_median_s"), 2),
                "latency_p95_s": fmt(s.get("latency_p95_s"), 2),
                "peak_memory_gb_max": fmt(bytes_to_gb(s.get("peak_rss_bytes_max")), 2),
                "retrieval_calls_mean": fmt(s.get("retrieval_calls_mean"), 3),
                "notes": spec.notes,
            }
        )
    write_csv(
        TABLES_DIR / "table_04_hotpot_main_results.csv",
        hotpot_main_rows,
        ["system", "model", "mode", "scope", "status", "queries", "answer_accuracy", "answer_f1", "evidence_recall", "supporting_doc_recall", "latency_median_s", "latency_p95_s", "peak_memory_gb_max", "retrieval_calls_mean", "notes"],
    )

    # Retry utility / agentic behavior
    retry_rows = []
    for key in ["legal_agentic_llama_3b", "legal_agentic_llama_8b", "legal_agentic_qwen_3b", "legal_agentic_qwen_7b", "hotpot_agentic_llama_3b", "hotpot_agentic_qwen_3b", "hotpot_agentic_llama_8b", "hotpot_agentic_qwen_7b"]:
        spec = next(s for s in run_specs if s.key == key)
        rows = run_rows[key]
        if not rows:
            retry_rows.append(
                {
                    "benchmark": spec.benchmark,
                    "system": spec.label,
                    "scope": spec.scope,
                    "status": spec.status,
                    "total_queries": 0,
                    "queries_with_retry": "",
                    "queries_without_retry": "",
                    "corrected_pass_selected": "",
                    "first_pass_selected_after_retry": "",
                    "single_pass_selected": "",
                    "selector_unknown": "",
                    "selector_parse_recovered_count": "",
                    "selector_parse_recovered_rate": "",
                    "selector_parse_fallback_count": "",
                    "selector_parse_fallback_rate": "",
                    "retrieval_calls_mean": "",
                    "iterations_mean": "",
                    "notes": spec.notes,
                }
            )
            continue
        selection_counts = Counter(r.get("answer_selection_source") for r in rows)
        parse_recovered_count = sum(1 for r in rows if r.get("answer_selection_reason") == "selector_parse_recovered")
        parse_fallback_count = sum(1 for r in rows if r.get("answer_selection_reason") == "selector_parse_fallback")
        retry_rows.append(
            {
                "benchmark": spec.benchmark,
                "system": spec.label,
                "scope": spec.scope,
                "status": spec.status,
                "total_queries": len(rows),
                "queries_with_retry": sum(1 for r in rows if (r.get("retrieval_calls") or 0) > 1),
                "queries_without_retry": sum(1 for r in rows if (r.get("retrieval_calls") or 0) <= 1),
                "corrected_pass_selected": selection_counts.get("corrected_pass", ""),
                "first_pass_selected_after_retry": selection_counts.get("first_pass", ""),
                "single_pass_selected": selection_counts.get("single_pass", ""),
                "selector_unknown": selection_counts.get("selector_unknown", ""),
                "selector_parse_recovered_count": parse_recovered_count,
                "selector_parse_recovered_rate": fmt(parse_recovered_count / len(rows), 3),
                "selector_parse_fallback_count": parse_fallback_count,
                "selector_parse_fallback_rate": fmt(parse_fallback_count / len(rows), 3),
                "retrieval_calls_mean": fmt(mean_num(rows, "retrieval_calls"), 3),
                "iterations_mean": fmt(mean_num(rows, "iterations"), 3),
                "notes": spec.notes,
            }
        )
    write_csv(
        TABLES_DIR / "table_05_retry_utility.csv",
        retry_rows,
        ["benchmark", "system", "scope", "status", "total_queries", "queries_with_retry", "queries_without_retry", "corrected_pass_selected", "first_pass_selected_after_retry", "single_pass_selected", "selector_unknown", "selector_parse_recovered_count", "selector_parse_recovered_rate", "selector_parse_fallback_count", "selector_parse_fallback_rate", "retrieval_calls_mean", "iterations_mean", "notes"],
    )

    # Legal per-subset results (Llama and Qwen comparisons)
    query_to_subset = load_legal_subset_map()
    legal_per_subset_rows = []
    subset_specs = [
        ("Llama", "legal_trad_llama_8b", "legal_agentic_llama_8b"),
        ("Qwen", "legal_trad_qwen_7b", "legal_agentic_qwen_7b"),
    ]
    for family, trad_key, agent_key in subset_specs:
        trad_rows = run_rows[trad_key]
        agent_rows = run_rows[agent_key]
        for subset in ["privacy_qa", "contractnli", "maud", "cuad"]:
            trad_subset = [r for r in trad_rows if query_to_subset.get(r.get("query", "")) == subset]
            agent_subset = [r for r in agent_rows if query_to_subset.get(r.get("query", "")) == subset]
            trad_ev = mean_bool(trad_subset, "evidence_recall")
            agent_ev = mean_bool(agent_subset, "evidence_recall")
            trad_topk = mean_bool(trad_subset, "topk_hit")
            agent_topk = mean_bool(agent_subset, "topk_hit")
            legal_per_subset_rows.append(
                {
                    "model_family": family,
                    "traditional_system": next(s for s in run_specs if s.key == trad_key).label,
                    "agentic_system": next(s for s in run_specs if s.key == agent_key).label,
                    "subset": subset,
                    "queries": len(trad_subset),
                    "traditional_evidence_recall": fmt(trad_ev),
                    "agentic_evidence_recall": fmt(agent_ev),
                    "delta_evidence_recall": fmt((agent_ev - trad_ev) if trad_ev is not None and agent_ev is not None else None),
                    "traditional_topk_hit_rate": fmt(trad_topk),
                    "agentic_topk_hit_rate": fmt(agent_topk),
                    "delta_topk_hit_rate": fmt((agent_topk - trad_topk) if trad_topk is not None and agent_topk is not None else None),
                }
            )
    write_csv(
        TABLES_DIR / "table_06_legal_per_subset_results.csv",
        legal_per_subset_rows,
        ["model_family", "traditional_system", "agentic_system", "subset", "queries", "traditional_evidence_recall", "agentic_evidence_recall", "delta_evidence_recall", "traditional_topk_hit_rate", "agentic_topk_hit_rate", "delta_topk_hit_rate"],
    )

    # Model sensitivity table
    model_rows = []
    for key in [
        "legal_trad_llama_3b",
        "legal_agentic_llama_3b",
        "legal_trad_llama_8b",
        "hotpot_trad_llama_3b",
        "hotpot_agentic_llama_3b",
        "legal_trad_qwen_3b",
        "legal_trad_qwen_7b",
        "legal_agentic_llama_8b",
        "legal_agentic_qwen_3b",
        "legal_agentic_qwen_7b",
        "hotpot_trad_llama_8b",
        "hotpot_trad_qwen_3b",
        "hotpot_trad_qwen_7b",
        "hotpot_agentic_qwen_3b",
        "hotpot_agentic_llama_8b",
        "hotpot_agentic_qwen_7b",
    ]:
        spec = next(s for s in run_specs if s.key == key)
        s = run_summary[key]
        model_rows.append(
            {
                "benchmark": spec.benchmark,
                "mode": spec.mode,
                "model": spec.model,
                "scope": spec.scope,
                "status": spec.status,
                "main_quality_metric": spec.main_quality_metric,
                "main_quality_value": fmt(s.get(spec.main_quality_value_key)),
                "latency_median_s": fmt(s.get("latency_median_s"), 2),
                "retrieval_calls_mean": fmt(s.get("retrieval_calls_mean"), 3),
                "evidence_recall": fmt(s.get("evidence_recall")),
                "answer_accuracy": fmt(s.get("answer_accuracy")),
                "notes": spec.notes,
            }
        )
    write_csv(
        TABLES_DIR / "table_07_model_sensitivity.csv",
        model_rows,
        ["benchmark", "mode", "model", "scope", "status", "main_quality_metric", "main_quality_value", "latency_median_s", "retrieval_calls_mean", "evidence_recall", "answer_accuracy", "notes"],
    )

    # Figure sources
    legal_figure_rows = [
        {
            "system": row["system"],
            "evidence_recall": row["evidence_recall"],
            "topk_hit_rate": row["topk_hit_rate"],
            "status": row["status"],
            "scope": row["scope"],
        }
        for row in legal_main_rows
    ]
    write_csv(
        FIGURES_DIR / "figure_02_legal_main_grouped_bar.csv",
        legal_figure_rows,
        ["system", "evidence_recall", "topk_hit_rate", "status", "scope"],
    )

    hotpot_answer_fig_rows = [
        {
            "system": row["system"],
            "answer_accuracy": row["answer_accuracy"],
            "answer_f1": row["answer_f1"],
            "status": row["status"],
            "scope": row["scope"],
        }
        for row in hotpot_main_rows
    ]
    write_csv(
        FIGURES_DIR / "figure_03_hotpot_answer_metrics.csv",
        hotpot_answer_fig_rows,
        ["system", "answer_accuracy", "answer_f1", "status", "scope"],
    )

    hotpot_retrieval_fig_rows = [
        {
            "system": row["system"],
            "evidence_recall": row["evidence_recall"],
            "supporting_doc_recall": row["supporting_doc_recall"],
            "status": row["status"],
            "scope": row["scope"],
        }
        for row in hotpot_main_rows
    ]
    write_csv(
        FIGURES_DIR / "figure_03_hotpot_retrieval_metrics.csv",
        hotpot_retrieval_fig_rows,
        ["system", "evidence_recall", "supporting_doc_recall", "status", "scope"],
    )

    latency_quality_rows = []
    for spec in run_specs:
        s = run_summary[spec.key]
        if not s:
            continue
        latency_quality_rows.append(
            {
                "benchmark": spec.benchmark,
                "system": spec.label,
                "model": spec.model,
                "mode": spec.mode,
                "scope": spec.scope,
                "status": spec.status,
                "latency_median_s": fmt(s.get("latency_median_s"), 2),
                "main_quality_metric": spec.main_quality_metric,
                "main_quality_value": fmt(s.get(spec.main_quality_value_key)),
            }
        )
    write_csv(
        FIGURES_DIR / "figure_04_latency_vs_quality.csv",
        latency_quality_rows,
        ["benchmark", "system", "model", "mode", "scope", "status", "latency_median_s", "main_quality_metric", "main_quality_value"],
    )

    subset_fig_rows = [
        {
            "model_family": row["model_family"],
            "subset": row["subset"],
            "traditional_system": row["traditional_system"],
            "agentic_system": row["agentic_system"],
            "delta_evidence_recall": row["delta_evidence_recall"],
            "delta_topk_hit_rate": row["delta_topk_hit_rate"],
        }
        for row in legal_per_subset_rows
    ]
    write_csv(
        FIGURES_DIR / "figure_05_legal_subset_deltas.csv",
        subset_fig_rows,
        ["model_family", "subset", "traditional_system", "agentic_system", "delta_evidence_recall", "delta_topk_hit_rate"],
    )

    decision_rows = []
    for key in ["hotpot_agentic_llama_3b", "hotpot_agentic_qwen_3b", "hotpot_agentic_llama_8b", "hotpot_agentic_qwen_7b"]:
        spec = next(s for s in run_specs if s.key == key)
        hotpot_agentic_rows = run_rows[key]
        selection_counts = Counter(r.get("answer_selection_source") or "none" for r in hotpot_agentic_rows)
        parse_recovered_count = sum(
            1 for r in hotpot_agentic_rows if r.get("answer_selection_reason") == "selector_parse_recovered"
        )
        parse_fallback_count = sum(
            1 for r in hotpot_agentic_rows if r.get("answer_selection_reason") == "selector_parse_fallback"
        )
        decision_rows.append(
            {
                "system": spec.label,
                "model_family": "Qwen" if "Qwen" in spec.label else "Llama",
                "single_pass": selection_counts.get("single_pass", 0),
                "first_pass": selection_counts.get("first_pass", 0),
                "corrected_pass": selection_counts.get("corrected_pass", 0),
                "selector_unknown": selection_counts.get("selector_unknown", 0),
                "selector_parse_recovered": parse_recovered_count,
                "selector_parse_recovered_rate": fmt(parse_recovered_count / len(hotpot_agentic_rows), 3),
                "selector_parse_fallback": parse_fallback_count,
                "selector_parse_fallback_rate": fmt(parse_fallback_count / len(hotpot_agentic_rows), 3),
            }
        )
    write_csv(
        FIGURES_DIR / "figure_06_hotpot_agentic_decision_outcomes.csv",
        decision_rows,
        ["system", "model_family", "single_pass", "first_pass", "corrected_pass", "selector_unknown", "selector_parse_recovered", "selector_parse_recovered_rate", "selector_parse_fallback", "selector_parse_fallback_rate"],
    )

    # Manifest / usage notes
    manifest = [
        "# Thesis Reporting Pack\n",
        "This folder contains first-draft CSV tables and figure source data generated from the current benchmark results.\n",
        "\n",
        "## Current Status\n",
        "- Core Llama runs are included.\n",
        "- Legal Llama 3B agentic full run is included as an extra model-choice experiment.\n",
        "- Legal Qwen 3B traditional full run is included as an extra model-choice experiment.\n",
        "- Legal Qwen traditional full run is included as an extra model-choice experiment.\n",
        "- Legal Qwen 3B agentic full run is included as an extra model-choice experiment.\n",
        "- Hotpot Qwen 3B traditional full run is included as an extra model-choice experiment.\n",
        "- Hotpot Qwen 7B traditional full run is included as an extra model-choice experiment.\n",
        "- Hotpot Qwen 3B agentic full run is included as an extra model-choice experiment.\n",
        "- Hotpot Qwen 7B agentic full run is included as an extra model-choice experiment.\n",
        "- Legal Qwen 7B agentic full run is included as an extra model-choice experiment.\n",
        "\n",
        "## Folders\n",
        f"- Tables: `{TABLES_DIR}`\n",
        f"- Figure sources: `{FIGURES_DIR}`\n",
        "\n",
        "## How To Refresh\n",
        f"Run: `python {Path(__file__).resolve()}`\n",
        "\n",
        "## Notes\n",
        "- Main tables are intended for the thesis body.\n",
        "- Preliminary Qwen rows are kept clearly marked so they do not get confused with finalized full comparisons.\n",
        "- Legal answer accuracy is intentionally not used as a main metric here because the current LegalBench-mini evaluation is retrieval-focused.\n",
    ]
    (OUT_ROOT / "README.md").write_text("".join(manifest), encoding="utf-8")

    print(f"Wrote reporting pack to {OUT_ROOT}")


if __name__ == "__main__":
    main()
