import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from src.config.defaults import LLAMA_GGUF_PATH_3B, LLAMA_GGUF_PATH_8B
from src.eval.datasets import iter_contractnli_cases_window
from src.eval.metrics import evidence_recall_multi_file
from src.rag import TraditionalRAG
from src.utils.memory import PeakMemory
from src.utils.timers import Timer


@dataclass
class SmokeRow:
    offset: int
    case_id: str
    variant_type: str
    query: str
    answer: str
    latency_s: float
    peak_rss_bytes: int
    iterations: int
    retrieval_calls: int
    topk_hit: bool
    evidence_recall: bool
    retrieval_latency_s: float
    generation_latency_s: float


def summarize(rows: list[SmokeRow]) -> dict:
    grouped = {}
    for variant in {r.variant_type for r in rows}:
        grp = [r for r in rows if r.variant_type == variant]
        grouped[variant] = {
            "count": len(grp),
            "latency_median_s": median(r.latency_s for r in grp),
            "iterations_mean": mean(r.iterations for r in grp),
            "topk_hit_rate": mean(1.0 if r.topk_hit else 0.0 for r in grp),
            "evidence_recall_mean": mean(1.0 if r.evidence_recall else 0.0 for r in grp),
        }
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", default="data/annotations/legalbench_query_robustness_smoke.csv")
    parser.add_argument("--model", choices=["3b", "8b"], default="3b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-jsonl", default="data/results/legalbench_query_robustness_smoke_traditional.jsonl")
    parser.add_argument("--output-summary", default="data/results/legalbench_query_robustness_smoke_traditional_summary.json")
    args = parser.parse_args()

    model_path = LLAMA_GGUF_PATH_3B if args.model == "3b" else LLAMA_GGUF_PATH_8B
    rag = TraditionalRAG(model_path=model_path)

    cases_by_offset = {}
    with open(args.pairs_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            off = int(row["offset"])
            if off not in cases_by_offset:
                cases_by_offset[off] = next(iter_contractnli_cases_window(offset=off, limit=1))

    rows_out: list[SmokeRow] = []
    with open(args.pairs_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            off = int(row["offset"])
            case = cases_by_offset[off]
            with PeakMemory() as mem, Timer() as timer:
                out = rag.run(row["query"], top_k=args.top_k)
            rows_out.append(
                SmokeRow(
                    offset=off,
                    case_id=row["case_id"],
                    variant_type=row["variant_type"],
                    query=row["query"],
                    answer=out.answer,
                    latency_s=timer.elapsed,
                    peak_rss_bytes=mem.peak_rss_bytes,
                    iterations=1,
                    retrieval_calls=1,
                    topk_hit=evidence_recall_multi_file(out.retrieved, case.gold_spans),
                    evidence_recall=evidence_recall_multi_file(out.used, case.gold_spans),
                    retrieval_latency_s=out.retrieval_latency_s,
                    generation_latency_s=out.generation_latency_s,
                )
            )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "by_variant": summarize(rows_out),
    }

    offsets = sorted({row.offset for row in rows_out})
    paired = []
    for off in offsets:
        orig = next(r for r in rows_out if r.offset == off and r.variant_type == "original")
        user = next(r for r in rows_out if r.offset == off and r.variant_type == "user_style")
        paired.append(
            {
                "offset": off,
                "case_id": orig.case_id,
                "latency_delta_user_minus_original_s": user.latency_s - orig.latency_s,
                "topk_changed": user.topk_hit != orig.topk_hit,
                "evidence_changed": user.evidence_recall != orig.evidence_recall,
            }
        )
    summary["pairwise"] = paired

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary\n")
    for variant, stats in summary["by_variant"].items():
        print(variant)
        for k, v in stats.items():
            print(f"  {k}: {v}")
    print(f"\nJSONL written to {output_jsonl}")
    print(f"Summary written to {output_summary}")


if __name__ == "__main__":
    main()
