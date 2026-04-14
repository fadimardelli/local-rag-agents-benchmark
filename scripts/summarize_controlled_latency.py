import argparse
import csv
import glob
import json
from pathlib import Path

from src.eval.harness import EvalResult, summarize


def _load_results(path: Path) -> list[EvalResult]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(EvalResult(**payload))
    return rows


def _write_key_value_csv(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in payload.items():
            writer.writerow([key, value])


def _write_run_rows_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise ValueError("No input files matched the provided glob")

    pooled_results = []
    run_rows = []

    for path in paths:
        results = _load_results(path)
        pooled_results.extend(results)
        summary = summarize(results)
        run_rows.append(
            {
                "source_jsonl": str(path),
                "run_id": results[0].run_id if results else "",
                "batch_id": results[0].batch_id if results else "",
                "count": len(results),
                "latency_median_s": summary.get("latency_median_s"),
                "latency_p95_s": summary.get("latency_p95_s"),
                "latency_max_s": summary.get("latency_max_s"),
                "retrieval_latency_mean_s": summary.get("retrieval_latency_mean_s"),
                "generation_latency_mean_s": summary.get("generation_latency_mean_s"),
                "retrieval_calls_mean": summary.get("retrieval_calls_mean"),
                "prompt_tokens_mean": summary.get("prompt_tokens_mean"),
                "answer_tokens_mean": summary.get("answer_tokens_mean"),
                "peak_rss_bytes_max": summary.get("peak_rss_bytes_max"),
                "batch_latency_drift_abs_s": summary.get("batch_latency_drift_abs_s"),
                "batch_latency_drift_pct": summary.get("batch_latency_drift_pct"),
            }
        )

    pooled_summary = summarize(pooled_results)
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(f"{output_prefix}_pooled_summary.json")
    csv_path = Path(f"{output_prefix}_pooled_summary.csv")
    runs_csv_path = Path(f"{output_prefix}_run_summaries.csv")

    json_path.write_text(
        json.dumps(
            {
                "input_glob": args.input_glob,
                "matched_files": [str(path) for path in paths],
                "pooled_summary": pooled_summary,
                "run_summaries": run_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_key_value_csv(pooled_summary, csv_path)
    _write_run_rows_csv(run_rows, runs_csv_path)

    print(f"Wrote pooled JSON summary to {json_path}")
    print(f"Wrote pooled CSV summary to {csv_path}")
    print(f"Wrote run-level CSV summary to {runs_csv_path}")


if __name__ == "__main__":
    main()
