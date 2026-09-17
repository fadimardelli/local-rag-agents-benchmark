from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

from final_result_specs import LOCAL_FINAL_ROOT, RUN_SPECS, RunSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = PROJECT_ROOT / "thesis"
OUT_ROOT = THESIS_ROOT / "reporting_pack"
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"
LOCAL_FINAL_DIR = PROJECT_ROOT / LOCAL_FINAL_ROOT
RUNPOD_COMPARISON_PATH = LOCAL_FINAL_DIR / "runpod_comparison.json"
LOCAL_INVENTORY_PATH = LOCAL_FINAL_DIR / "inventory.json"
LOCAL_README_PATH = LOCAL_FINAL_DIR / "README.md"


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


def pstdev_num(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return pstdev(vals) if vals else None


def median_num(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return median(vals) if vals else None


def p95_num(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return percentile(vals, 0.95) if vals else None


def max_num(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return max(vals) if vals else None


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


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers = [str(r.get("answer", "")) for r in rows]
    retry_rate = None
    if rows:
        retry_rate = sum(1 for r in rows if (r.get("iterations") or 1) > 1) / len(rows)
    return {
        "queries": len(rows),
        "answer_accuracy": mean_bool(rows, "correct"),
        "answer_f1": mean_num(rows, "answer_f1"),
        "evidence_recall": mean_bool(rows, "evidence_recall"),
        "topk_hit_rate": mean_bool(rows, "topk_hit"),
        "supporting_doc_recall": mean_num(rows, "supporting_doc_recall"),
        "retry_rate": retry_rate,
        "latency_median_s": median_num(rows, "latency_s"),
        "latency_p95_s": p95_num(rows, "latency_s"),
        "latency_max_s": max_num(rows, "latency_s"),
        "peak_rss_bytes_max": max_num(rows, "peak_rss_bytes"),
        "retrieval_calls_mean": mean_num(rows, "retrieval_calls"),
        "retrieval_calls_std": pstdev_num(rows, "retrieval_calls"),
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


def condition_order(spec: RunSpec) -> tuple[int, int, int]:
    bench_rank = 0 if spec.benchmark == "LegalBench-mini" else 1
    family_rank = 0 if spec.family == "Qwen" else 1
    mode_rank = 0 if spec.mode == "traditional" else 1
    return (bench_rank, family_rank, spec.size_b * 10 + mode_rank)


def primary_quality_metric(spec: RunSpec) -> tuple[str, str]:
    if spec.benchmark == "LegalBench-mini":
        return ("Evidence Recall", "evidence_recall")
    return ("Answer F1", "answer_f1")


def compare_metrics_for(spec: RunSpec) -> list[str]:
    if spec.benchmark == "LegalBench-mini":
        return ["evidence_recall", "topk_hit_rate", "retry_rate", "iterations_mean", "retrieval_calls_mean"]
    return [
        "answer_accuracy",
        "answer_f1",
        "supporting_doc_recall",
        "evidence_recall",
        "topk_hit_rate",
        "retry_rate",
        "iterations_mean",
        "retrieval_calls_mean",
    ]


def build_subset_summary(rows: list[dict[str, Any]], query_to_subset: dict[str, str]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[query_to_subset.get(row.get("query", ""), "unknown")].append(row)
    out: dict[str, dict[str, float | None]] = {}
    for subset in ["privacy_qa", "contractnli", "maud", "cuad"]:
        subset_rows = grouped.get(subset, [])
        out[subset] = {
            "queries": len(subset_rows),
            "evidence_recall": mean_bool(subset_rows, "evidence_recall"),
            "topk_hit_rate": mean_bool(subset_rows, "topk_hit"),
            "retry_rate": (sum(1 for r in subset_rows if (r.get("iterations") or 1) > 1) / len(subset_rows)) if subset_rows else None,
        }
    return out


def build_runpod_comparison(specs: list[RunSpec], local_summary: dict[str, dict[str, Any]], pod_summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparison = []
    for spec in specs:
        if spec.pod_glob is None:
            continue
        pod = pod_summary.get(spec.key)
        if not pod:
            continue
        metrics = {}
        for metric in compare_metrics_for(spec):
            local_val = local_summary[spec.key].get(metric)
            pod_val = pod.get(metric)
            delta = None if local_val is None or pod_val is None else local_val - pod_val
            metrics[metric] = {
                "local": local_val,
                "runpod": pod_val,
                "delta": delta,
                "abs_delta": None if delta is None else abs(delta),
            }
        comparison.append(
            {
                "benchmark": spec.benchmark.lower().replace("-", "_").replace(" ", "_"),
                "condition": spec.key.replace("legal_", "").replace("hotpot_", ""),
                "local_dir": str((PROJECT_ROOT / spec.local_glob.split("*.jsonl")[0]).resolve()),
                "runpod_glob": str((PROJECT_ROOT / spec.pod_glob).resolve()),
                "metrics": metrics,
            }
        )
    return comparison


def refresh_local_inventory(specs: list[RunSpec]) -> None:
    included = []
    for spec in specs:
        folder = PROJECT_ROOT / spec.local_glob.split("*.jsonl")[0]
        files = sorted(folder.glob("*"))
        latest_ts = None
        if files:
            latest_ts = max(p.stat().st_mtime for p in files)
        included.append(
            {
                "key": spec.key,
                "benchmark": spec.benchmark,
                "mode": spec.mode,
                "model": spec.model,
                "label": spec.label,
                "folder": str(folder),
                "file_count": len(files),
                "latest_source_timestamp": "" if latest_ts is None else datetime.fromtimestamp(latest_ts, tz=timezone.utc).isoformat(),
                "notes": spec.notes,
            }
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Curated local final-result inventory after all corrected local reruns completed.",
        "included": included,
        "excluded_pending_rerun": [],
    }
    LOCAL_INVENTORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Local Latest Usable Results (2026-04-17)\n\n",
        "This folder contains the curated local final-result artifacts used as the canonical source for thesis tables and figures.\n\n",
        "All 16 benchmark conditions are now present in corrected local form.\n\n",
        "Included conditions:\n",
    ]
    for item in included:
        lines.append(
            f"- {item['benchmark']} / {item['mode']} / {item['model']}: {item['file_count']} files; latest timestamp {item['latest_source_timestamp'] or 'unknown'}\n"
        )
    LOCAL_README_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for stale in TABLES_DIR.glob("*.csv"):
        stale.unlink()
    for stale in FIGURES_DIR.glob("*.csv"):
        stale.unlink()

    query_to_subset = load_legal_subset_map()

    run_rows = {spec.key: load_jsonl_rows(spec.local_glob) for spec in RUN_SPECS}
    run_summary = {spec.key: summarize_run(run_rows[spec.key]) for spec in RUN_SPECS}
    pod_rows = {spec.key: load_jsonl_rows(spec.pod_glob) if spec.pod_glob else [] for spec in RUN_SPECS}
    pod_summary = {spec.key: summarize_run(pod_rows[spec.key]) if pod_rows[spec.key] else {} for spec in RUN_SPECS}
    subset_summary = {
        spec.key: build_subset_summary(run_rows[spec.key], query_to_subset)
        for spec in RUN_SPECS
        if spec.benchmark == "LegalBench-mini"
    }

    refresh_local_inventory(RUN_SPECS)
    runpod_comparison = build_runpod_comparison(RUN_SPECS, run_summary, pod_summary)
    RUNPOD_COMPARISON_PATH.write_text(json.dumps(runpod_comparison, indent=2), encoding="utf-8")

    ordered_specs = sorted(RUN_SPECS, key=condition_order)

    # Canonical all-condition summary
    condition_summary_rows = []
    for spec in ordered_specs:
        s = run_summary[spec.key]
        primary_name, primary_key = primary_quality_metric(spec)
        condition_summary_rows.append(
            {
                "benchmark": spec.benchmark,
                "family": spec.family,
                "size_b": spec.size_b,
                "model": spec.model,
                "mode": spec.mode,
                "label": spec.label,
                "scope": spec.scope,
                "queries": s.get("queries", 0),
                "primary_quality_metric": primary_name,
                "primary_quality_value": fmt(s.get(primary_key)),
                "answer_accuracy": fmt(s.get("answer_accuracy")),
                "answer_f1": fmt(s.get("answer_f1")),
                "supporting_doc_recall": fmt(s.get("supporting_doc_recall")),
                "evidence_recall": fmt(s.get("evidence_recall")),
                "topk_hit_rate": fmt(s.get("topk_hit_rate")),
                "retry_rate": fmt(s.get("retry_rate")),
                "iterations_mean": fmt(s.get("iterations_mean")),
                "retrieval_calls_mean": fmt(s.get("retrieval_calls_mean")),
                "latency_median_s": fmt(s.get("latency_median_s"), 2),
                "latency_p95_s": fmt(s.get("latency_p95_s"), 2),
                "peak_memory_gb_max": fmt(bytes_to_gb(s.get("peak_rss_bytes_max")), 2),
                "retrieval_latency_mean_s": fmt(s.get("retrieval_latency_mean_s"), 3),
                "generation_latency_mean_s": fmt(s.get("generation_latency_mean_s"), 3),
                "notes": spec.notes,
            }
        )
    write_csv(
        TABLES_DIR / "condition_summary.csv",
        condition_summary_rows,
        list(condition_summary_rows[0].keys()),
    )

    # Main paper tables
    legal_rows = []
    for spec in ordered_specs:
        if spec.benchmark != "LegalBench-mini":
            continue
        s = run_summary[spec.key]
        legal_rows.append(
            {
                "model": spec.model,
                "family": spec.family,
                "size_b": spec.size_b,
                "mode": spec.mode,
                "evidence_recall": fmt(s.get("evidence_recall")),
                "topk_hit_rate": fmt(s.get("topk_hit_rate")),
                "retry_rate": fmt(s.get("retry_rate")),
                "iterations_mean": fmt(s.get("iterations_mean")),
            }
        )
    write_csv(
        TABLES_DIR / "table_01_overall_legal_results.csv",
        legal_rows,
        ["model", "family", "size_b", "mode", "evidence_recall", "topk_hit_rate", "retry_rate", "iterations_mean"],
    )

    legal_subset_wide_rows = []
    for spec in ordered_specs:
        if spec.benchmark != "LegalBench-mini":
            continue
        subsets = subset_summary[spec.key]
        legal_subset_wide_rows.append(
            {
                "model": spec.model,
                "family": spec.family,
                "size_b": spec.size_b,
                "mode": spec.mode,
                "privacy_qa_evidence_recall": fmt(subsets["privacy_qa"]["evidence_recall"]),
                "contractnli_evidence_recall": fmt(subsets["contractnli"]["evidence_recall"]),
                "maud_evidence_recall": fmt(subsets["maud"]["evidence_recall"]),
                "cuad_evidence_recall": fmt(subsets["cuad"]["evidence_recall"]),
            }
        )
    write_csv(
        TABLES_DIR / "table_02_legal_subset_results.csv",
        legal_subset_wide_rows,
        [
            "model",
            "family",
            "size_b",
            "mode",
            "privacy_qa_evidence_recall",
            "contractnli_evidence_recall",
            "maud_evidence_recall",
            "cuad_evidence_recall",
        ],
    )

    hotpot_quality_rows = []
    hotpot_activation_rows = []
    for spec in ordered_specs:
        if spec.benchmark != "HotpotQA":
            continue
        s = run_summary[spec.key]
        base = {"model": spec.model, "family": spec.family, "size_b": spec.size_b, "mode": spec.mode}
        hotpot_quality_rows.append(
            {
                **base,
                "answer_accuracy": fmt(s.get("answer_accuracy")),
                "answer_f1": fmt(s.get("answer_f1")),
                "supporting_doc_recall": fmt(s.get("supporting_doc_recall")),
            }
        )
        hotpot_activation_rows.append(
            {
                **base,
                "retry_rate": fmt(s.get("retry_rate")),
                "iterations_mean": fmt(s.get("iterations_mean")),
            }
        )
    write_csv(
        TABLES_DIR / "table_03_hotpot_quality_results.csv",
        hotpot_quality_rows,
        ["model", "family", "size_b", "mode", "answer_accuracy", "answer_f1", "supporting_doc_recall"],
    )
    write_csv(
        TABLES_DIR / "table_04_hotpot_activation_results.csv",
        hotpot_activation_rows,
        ["model", "family", "size_b", "mode", "retry_rate", "iterations_mean"],
    )

    cross_legal_rows = []
    cross_hotpot_rows = []
    appendix_cross_rows = []
    for spec in ordered_specs:
        pod = pod_summary.get(spec.key) or None
        if not pod:
            continue
        local = run_summary[spec.key]
        if spec.benchmark == "LegalBench-mini":
            cross_legal_rows.append(
                {
                    "model": spec.model,
                    "family": spec.family,
                    "size_b": spec.size_b,
                    "mode": spec.mode,
                    "local_evidence_recall": fmt(local.get("evidence_recall")),
                    "pod_evidence_recall": fmt(pod.get("evidence_recall")),
                    "delta_evidence_recall": fmt((local.get("evidence_recall") or 0) - (pod.get("evidence_recall") or 0)),
                    "local_topk_hit_rate": fmt(local.get("topk_hit_rate")),
                    "pod_topk_hit_rate": fmt(pod.get("topk_hit_rate")),
                    "delta_topk_hit_rate": fmt((local.get("topk_hit_rate") or 0) - (pod.get("topk_hit_rate") or 0)),
                    "local_retry_rate": fmt(local.get("retry_rate")),
                    "pod_retry_rate": fmt(pod.get("retry_rate")),
                }
            )
        else:
            cross_hotpot_rows.append(
                {
                    "model": spec.model,
                    "family": spec.family,
                    "size_b": spec.size_b,
                    "mode": spec.mode,
                    "local_answer_accuracy": fmt(local.get("answer_accuracy")),
                    "pod_answer_accuracy": fmt(pod.get("answer_accuracy")),
                    "delta_answer_accuracy": fmt((local.get("answer_accuracy") or 0) - (pod.get("answer_accuracy") or 0)),
                    "local_answer_f1": fmt(local.get("answer_f1")),
                    "pod_answer_f1": fmt(pod.get("answer_f1")),
                    "delta_answer_f1": fmt((local.get("answer_f1") or 0) - (pod.get("answer_f1") or 0)),
                    "local_supporting_doc_recall": fmt(local.get("supporting_doc_recall")),
                    "pod_supporting_doc_recall": fmt(pod.get("supporting_doc_recall")),
                    "delta_supporting_doc_recall": fmt((local.get("supporting_doc_recall") or 0) - (pod.get("supporting_doc_recall") or 0)),
                    "local_retry_rate": fmt(local.get("retry_rate")),
                    "pod_retry_rate": fmt(pod.get("retry_rate")),
                }
            )
        for metric in compare_metrics_for(spec):
            local_val = local.get(metric)
            pod_val = pod.get(metric)
            delta = None if local_val is None or pod_val is None else local_val - pod_val
            appendix_cross_rows.append(
                {
                    "benchmark": spec.benchmark,
                    "model": spec.model,
                    "family": spec.family,
                    "size_b": spec.size_b,
                    "mode": spec.mode,
                    "metric": metric,
                    "local_value": fmt(local_val),
                    "pod_value": fmt(pod_val),
                    "delta": fmt(delta),
                    "abs_delta": fmt(None if delta is None else abs(delta)),
                }
            )
    write_csv(
        TABLES_DIR / "table_05a_cross_environment_legal.csv",
        cross_legal_rows,
        list(cross_legal_rows[0].keys()),
    )
    write_csv(
        TABLES_DIR / "table_05b_cross_environment_hotpot.csv",
        cross_hotpot_rows,
        list(cross_hotpot_rows[0].keys()),
    )

    # Appendix / supporting tables
    efficiency_rows = []
    for spec in ordered_specs:
        s = run_summary[spec.key]
        efficiency_rows.append(
            {
                "benchmark": spec.benchmark,
                "model": spec.model,
                "family": spec.family,
                "size_b": spec.size_b,
                "mode": spec.mode,
                "latency_median_s": fmt(s.get("latency_median_s"), 2),
                "latency_p95_s": fmt(s.get("latency_p95_s"), 2),
                "peak_memory_gb_max": fmt(bytes_to_gb(s.get("peak_rss_bytes_max")), 2),
                "prompt_tokens_mean": fmt(s.get("prompt_tokens_mean"), 1),
                "answer_tokens_mean": fmt(s.get("answer_tokens_mean"), 1),
                "retrieval_latency_mean_s": fmt(s.get("retrieval_latency_mean_s"), 3),
                "generation_latency_mean_s": fmt(s.get("generation_latency_mean_s"), 3),
                "retrieval_calls_mean": fmt(s.get("retrieval_calls_mean"), 3),
            }
        )
    write_csv(
        TABLES_DIR / "appendix_01_efficiency_summary.csv",
        efficiency_rows,
        list(efficiency_rows[0].keys()),
    )

    legal_subset_long_rows = []
    for spec in ordered_specs:
        if spec.benchmark != "LegalBench-mini":
            continue
        for subset, vals in subset_summary[spec.key].items():
            legal_subset_long_rows.append(
                {
                    "model": spec.model,
                    "family": spec.family,
                    "size_b": spec.size_b,
                    "mode": spec.mode,
                    "subset": subset,
                    "queries": vals["queries"],
                    "evidence_recall": fmt(vals["evidence_recall"]),
                    "topk_hit_rate": fmt(vals["topk_hit_rate"]),
                    "retry_rate": fmt(vals["retry_rate"]),
                }
            )
    write_csv(
        TABLES_DIR / "appendix_02_legal_full_subset_metrics.csv",
        legal_subset_long_rows,
        list(legal_subset_long_rows[0].keys()),
    )

    hotpot_secondary_rows = []
    for spec in ordered_specs:
        if spec.benchmark != "HotpotQA":
            continue
        s = run_summary[spec.key]
        hotpot_secondary_rows.append(
            {
                "model": spec.model,
                "family": spec.family,
                "size_b": spec.size_b,
                "mode": spec.mode,
                "evidence_recall": fmt(s.get("evidence_recall")),
                "topk_hit_rate": fmt(s.get("topk_hit_rate")),
                "retrieval_calls_mean": fmt(s.get("retrieval_calls_mean")),
                "iterations_mean": fmt(s.get("iterations_mean")),
            }
        )
    write_csv(
        TABLES_DIR / "appendix_03_hotpot_secondary_retrieval.csv",
        hotpot_secondary_rows,
        list(hotpot_secondary_rows[0].keys()),
    )
    write_csv(
        TABLES_DIR / "appendix_04_cross_environment_full_deltas.csv",
        appendix_cross_rows,
        list(appendix_cross_rows[0].keys()),
    )

    # Figure source files
    heatmap_rows = []
    for spec in ordered_specs:
        if spec.benchmark != "LegalBench-mini":
            continue
        for subset, vals in subset_summary[spec.key].items():
            heatmap_rows.append(
                {
                    "system": spec.label,
                    "family": spec.family,
                    "size_b": spec.size_b,
                    "mode": spec.mode,
                    "subset": subset,
                    "evidence_recall": fmt(vals["evidence_recall"]),
                }
            )
    write_csv(
        FIGURES_DIR / "figure_01_legal_subset_heatmap.csv",
        heatmap_rows,
        list(heatmap_rows[0].keys()),
    )

    hotpot_quality_fig_rows = []
    for spec in ordered_specs:
        if spec.benchmark != "HotpotQA":
            continue
        s = run_summary[spec.key]
        for metric in ["answer_accuracy", "answer_f1", "supporting_doc_recall"]:
            hotpot_quality_fig_rows.append(
                {
                    "system": spec.label,
                    "family": spec.family,
                    "size_b": spec.size_b,
                    "mode": spec.mode,
                    "metric": metric,
                    "value": fmt(s.get(metric)),
                }
            )
    write_csv(
        FIGURES_DIR / "figure_02_hotpot_quality_comparison.csv",
        hotpot_quality_fig_rows,
        list(hotpot_quality_fig_rows[0].keys()),
    )

    activation_rows = []
    for spec in ordered_specs:
        if spec.mode != "agentic":
            continue
        s = run_summary[spec.key]
        activation_rows.append(
            {
                "benchmark": spec.benchmark,
                "system": spec.label,
                "family": spec.family,
                "size_b": spec.size_b,
                "retry_rate": fmt(s.get("retry_rate")),
                "iterations_mean": fmt(s.get("iterations_mean")),
                "retrieval_calls_mean": fmt(s.get("retrieval_calls_mean")),
                "retrieval_calls_std": fmt(s.get("retrieval_calls_std")),
            }
        )
    write_csv(
        FIGURES_DIR / "figure_03_corrective_activation.csv",
        activation_rows,
        list(activation_rows[0].keys()),
    )

    tradeoff_rows = []
    for spec in ordered_specs:
        s = run_summary[spec.key]
        primary_name, primary_key = primary_quality_metric(spec)
        tradeoff_rows.append(
            {
                "benchmark": spec.benchmark,
                "system": spec.label,
                "family": spec.family,
                "size_b": spec.size_b,
                "mode": spec.mode,
                "latency_median_s": fmt(s.get("latency_median_s"), 2),
                "quality_metric": primary_name,
                "quality_value": fmt(s.get(primary_key)),
            }
        )
    write_csv(
        FIGURES_DIR / "figure_04_quality_cost_tradeoff.csv",
        tradeoff_rows,
        list(tradeoff_rows[0].keys()),
    )

    agreement_rows = []
    for spec in ordered_specs:
        pod = pod_summary.get(spec.key) or None
        if not pod:
            continue
        local = run_summary[spec.key]
        for metric in compare_metrics_for(spec):
            local_val = local.get(metric)
            pod_val = pod.get(metric)
            if local_val is None or pod_val is None:
                continue
            agreement_rows.append(
                {
                    "benchmark": spec.benchmark,
                    "system": spec.label,
                    "family": spec.family,
                    "size_b": spec.size_b,
                    "mode": spec.mode,
                    "metric": metric,
                    "local_value": fmt(local_val),
                    "pod_value": fmt(pod_val),
                    "abs_delta": fmt(abs(local_val - pod_val)),
                }
            )
    write_csv(
        FIGURES_DIR / "figure_05_local_vs_pod_agreement.csv",
        agreement_rows,
        list(agreement_rows[0].keys()),
    )

    latency_split_rows = []
    for spec in ordered_specs:
        s = run_summary[spec.key]
        latency_split_rows.append(
            {
                "benchmark": spec.benchmark,
                "system": spec.label,
                "family": spec.family,
                "size_b": spec.size_b,
                "mode": spec.mode,
                "retrieval_latency_mean_s": fmt(s.get("retrieval_latency_mean_s"), 3),
                "generation_latency_mean_s": fmt(s.get("generation_latency_mean_s"), 3),
            }
        )
    write_csv(
        FIGURES_DIR / "figure_06_latency_decomposition.csv",
        latency_split_rows,
        list(latency_split_rows[0].keys()),
    )

    manifest = [
        "# Thesis Reporting Pack\n\n",
        "This folder contains refreshed table-ready and figure-ready CSV exports built from the final local result set.\n\n",
        "## Main Text Tables\n",
        "- `tables/table_01_overall_legal_results.csv`\n",
        "- `tables/table_02_legal_subset_results.csv`\n",
        "- `tables/table_03_hotpot_quality_results.csv`\n",
        "- `tables/table_04_hotpot_activation_results.csv`\n",
        "- `tables/table_05a_cross_environment_legal.csv`\n",
        "- `tables/table_05b_cross_environment_hotpot.csv`\n",
        "\n## Figure Source Files\n",
        "- `figures/figure_01_legal_subset_heatmap.csv`\n",
        "- `figures/figure_02_hotpot_quality_comparison.csv`\n",
        "- `figures/figure_03_corrective_activation.csv`\n",
        "- `figures/figure_04_quality_cost_tradeoff.csv`\n",
        "- `figures/figure_05_local_vs_pod_agreement.csv`\n",
        "\n## Supporting Files\n",
        "- `tables/condition_summary.csv`: canonical all-condition summary\n",
        "- `tables/appendix_01_efficiency_summary.csv`\n",
        "- `tables/appendix_02_legal_full_subset_metrics.csv`\n",
        "- `tables/appendix_03_hotpot_secondary_retrieval.csv`\n",
        "- `tables/appendix_04_cross_environment_full_deltas.csv`\n",
        f"\nRefreshed from local data under `{LOCAL_FINAL_DIR}`.\n",
    ]
    (OUT_ROOT / "README.md").write_text("".join(manifest), encoding="utf-8")

    print(f"Wrote reporting pack to {OUT_ROOT}")
    print(f"Updated local inventory at {LOCAL_INVENTORY_PATH}")
    print(f"Updated local/pod comparison at {RUNPOD_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
