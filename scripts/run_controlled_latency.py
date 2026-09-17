import argparse
import json
import platform
from pathlib import Path
from typing import Iterable, List

from src.config.defaults import (
    HYBRID_BM25_TOP_K,
    HYBRID_DENSE_TOP_K,
    HYBRID_RRF_K,
    LLAMA_GGUF_PATH_3B,
    LLAMA_GGUF_PATH_8B,
    QUERY_TRANSFORM_MODE,
    RERANK_CANDIDATE_K,
    RERANK_ENABLED,
    RERANK_MODEL_NAME,
    RETRIEVAL_MODE,
)
from src.eval.datasets import (
    ContractNLITestCase,
    HotpotQACase,
    iter_hotpotqa_cases,
    iter_legalbench_mini_balanced_cases_window,
    iter_legalbench_mini_cases_window,
)
from src.eval.harness import (
    run_eval,
    run_eval_hotpotqa,
    select_summary_view,
    summarize,
    write_results,
    write_results_csv,
    write_summary_csv,
)


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "benchmark" not in payload or "cases" not in payload:
        raise ValueError(f"Invalid manifest: {path}")
    return payload


def _slice_manifest_cases(payload: dict, manifest_offset: int, manifest_limit: int | None) -> list[dict]:
    cases = payload["cases"]
    start = manifest_offset
    end = None if manifest_limit is None else manifest_offset + manifest_limit
    selected = cases[start:end]
    if not selected:
        raise ValueError("Selected manifest slice is empty")
    return selected


def _apply_sweep_order(cases: list[dict], sweep_order: str) -> list[dict]:
    if sweep_order == "canonical":
        return cases
    if sweep_order == "reversed":
        return list(reversed(cases))
    raise ValueError(f"Unsupported sweep order: {sweep_order}")


def _legal_cases(benchmark: str, manifest_cases: list[dict]) -> List[ContractNLITestCase]:
    iterator_fn = (
        iter_legalbench_mini_balanced_cases_window
        if benchmark == "legalbench_mini_balanced"
        else iter_legalbench_mini_cases_window
    )
    selected = []
    for item in manifest_cases:
        offset = int(item["dataset_offset"])
        selected.append(next(iterator_fn(offset=offset, limit=1)))
    return selected


def _hotpot_cases(manifest_cases: list[dict]) -> List[HotpotQACase]:
    selected = []
    for item in manifest_cases:
        offset = int(item["dataset_offset"])
        selected.append(next(iter_hotpotqa_cases(offset=offset, limit=1)))
    return selected


def _default_model_path(model: str) -> Path:
    return LLAMA_GGUF_PATH_3B if model == "3b" else LLAMA_GGUF_PATH_8B


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["traditional", "agentic"], required=True)
    parser.add_argument("--manifest-offset", type=int, default=0)
    parser.add_argument("--manifest-limit", type=int, default=20)
    parser.add_argument("--model", choices=["8b", "3b"], default="8b")
    parser.add_argument("--model-label", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--repeat-label", type=str, default="")
    parser.add_argument("--study-stage", choices=["calibration", "final"], default="calibration")
    parser.add_argument("--sweep-order", choices=["canonical", "reversed"], default="canonical")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/Users/fadimardelli/local-rag-agents-benchmark/data/results/controlled_latency"),
    )
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--trace-agentic", action="store_true")
    parser.add_argument("--cooldown-seconds", type=int, default=120)
    parser.add_argument("--power-source", choices=["ac", "battery"], default="ac")
    parser.add_argument("--lid-state", choices=["open", "closed"], default="open")
    parser.add_argument("--low-power-mode", choices=["on", "off"], default="off")
    parser.add_argument("--background-load", choices=["idle", "light", "heavy"], default="idle")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--summary-view", choices=["all", "thesis", "calibration"], default="all")
    parser.add_argument("--query-transform-mode", choices=["none", "hyde"], default=QUERY_TRANSFORM_MODE)
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default=RETRIEVAL_MODE)
    parser.add_argument("--hybrid-dense-top-k", type=int, default=HYBRID_DENSE_TOP_K)
    parser.add_argument("--hybrid-bm25-top-k", type=int, default=HYBRID_BM25_TOP_K)
    parser.add_argument("--hybrid-rrf-k", type=int, default=HYBRID_RRF_K)
    parser.add_argument("--rerank-enabled", action="store_true", default=RERANK_ENABLED)
    parser.add_argument("--rerank-candidate-k", type=int, default=RERANK_CANDIDATE_K)
    parser.add_argument("--rerank-model-name", type=str, default=RERANK_MODEL_NAME)
    args = parser.parse_args()

    payload = _load_manifest(args.manifest)
    manifest_cases = _slice_manifest_cases(payload, args.manifest_offset, args.manifest_limit)
    manifest_cases = _apply_sweep_order(manifest_cases, args.sweep_order)
    benchmark = payload["benchmark"]
    model_path = Path(args.model_path) if args.model_path else _default_model_path(args.model)

    if benchmark in {"legalbench_mini", "legalbench_mini_balanced"}:
        cases = _legal_cases(benchmark, manifest_cases)
        results = run_eval(
            cases,
            mode=args.mode,
            warmup=args.warmup,
            model_path=model_path,
            run_id=args.run_id,
            batch_id=args.repeat_label or f"manoff{args.manifest_offset}_manlim{len(manifest_cases)}",
            trace_agentic=args.trace_agentic,
            top_k=args.top_k,
            query_transform_mode=args.query_transform_mode,
            retriever_kwargs={
                "retrieval_mode": args.retrieval_mode,
                "hybrid_dense_top_k": args.hybrid_dense_top_k,
                "hybrid_bm25_top_k": args.hybrid_bm25_top_k,
                "hybrid_rrf_k": args.hybrid_rrf_k,
                "rerank_enabled": args.rerank_enabled,
                "rerank_candidate_k": args.rerank_candidate_k,
                "rerank_model_name": args.rerank_model_name,
            },
        )
    elif benchmark == "hotpotqa":
        cases = _hotpot_cases(manifest_cases)
        results = run_eval_hotpotqa(
            cases,
            mode=args.mode,
            warmup=args.warmup,
            model_path=model_path,
            run_id=args.run_id,
            batch_id=args.repeat_label or f"manoff{args.manifest_offset}_manlim{len(manifest_cases)}",
            top_k=args.top_k,
            trace_agentic=args.trace_agentic,
        )
    else:
        raise ValueError(f"Unsupported manifest benchmark: {benchmark}")

    summary = select_summary_view(summarize(results), args.summary_view)

    benchmark_dir = args.results_dir / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.run_id}_manoff{args.manifest_offset}_manlim{len(manifest_cases)}"
    if args.repeat_label:
        suffix = f"{suffix}_{args.repeat_label}"

    jsonl_path = benchmark_dir / f"{suffix}.jsonl"
    csv_path = benchmark_dir / f"{suffix}.csv"
    summary_path = benchmark_dir / f"{suffix}_summary.csv"
    metadata_path = benchmark_dir / f"{suffix}_metadata.json"

    write_results(results, jsonl_path)
    write_results_csv(results, csv_path)
    write_summary_csv(summary, summary_path)
    metadata_path.write_text(
        json.dumps(
            {
                "manifest_path": str(args.manifest),
                "benchmark": benchmark,
                "mode": args.mode,
                "run_id": args.run_id,
                "repeat_label": args.repeat_label,
                "study_stage": args.study_stage,
                "sweep_order": args.sweep_order,
                "model_path": str(model_path),
                "model_label": args.model_label or args.model,
                "manifest_offset": args.manifest_offset,
                "manifest_limit": len(manifest_cases),
                "warmup": args.warmup,
                "trace_agentic": args.trace_agentic,
                "top_k": args.top_k,
                "summary_view": args.summary_view,
                "query_transform_mode": args.query_transform_mode,
                "retrieval_mode": args.retrieval_mode,
                "hybrid_dense_top_k": args.hybrid_dense_top_k,
                "hybrid_bm25_top_k": args.hybrid_bm25_top_k,
                "hybrid_rrf_k": args.hybrid_rrf_k,
                "rerank_enabled": args.rerank_enabled,
                "rerank_candidate_k": args.rerank_candidate_k,
                "rerank_model_name": args.rerank_model_name,
                "cooldown_seconds": args.cooldown_seconds,
                "power_source": args.power_source,
                "lid_state": args.lid_state,
                "low_power_mode": args.low_power_mode,
                "background_load": args.background_load,
                "execution_environment": {
                    "hostname": platform.node(),
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                "selected_dataset_offsets": [int(item["dataset_offset"]) for item in manifest_cases],
                "selected_queries": [item.get("query", "") for item in manifest_cases],
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote results to {jsonl_path}")
    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
