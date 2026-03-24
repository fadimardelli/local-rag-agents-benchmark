import argparse
import csv
from pathlib import Path
from typing import Dict, List


def load_summary_csv(path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["metric"]] = row["value"]
    return rows


def as_float(rows: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = rows.get(key)
    if value is None or value == "":
        return default
    return float(value)


def summarize_batch(path: Path) -> Dict[str, float]:
    rows = load_summary_csv(path)
    return {
        "latency_median_s": as_float(rows, "latency_median_s"),
        "latency_p95_s": as_float(rows, "latency_p95_s"),
        "batch_first_k_latency_median_s": as_float(rows, "batch_first_k_latency_median_s"),
        "batch_first_third_latency_median_s": as_float(rows, "batch_first_third_latency_median_s"),
        "batch_last_third_latency_median_s": as_float(rows, "batch_last_third_latency_median_s"),
        "batch_latency_drift_abs_s": as_float(rows, "batch_latency_drift_abs_s"),
        "batch_latency_drift_pct": as_float(rows, "batch_latency_drift_pct"),
        "timeout_rate": as_float(rows, "timeout_rate"),
        "error_rate": as_float(rows, "error_rate"),
        "queries_in_batch": as_float(rows, "queries_in_batch"),
    }


def aggregate(paths: List[Path]) -> Dict[str, float]:
    batches = [summarize_batch(path) for path in paths]
    if not batches:
        return {}

    def mean(key: str) -> float:
        return sum(batch[key] for batch in batches) / len(batches)

    return {
        "num_batches": float(len(batches)),
        "queries_total": float(sum(batch["queries_in_batch"] for batch in batches)),
        "latency_median_mean_s": mean("latency_median_s"),
        "latency_p95_mean_s": mean("latency_p95_s"),
        "first_k_latency_mean_s": mean("batch_first_k_latency_median_s"),
        "first_third_latency_mean_s": mean("batch_first_third_latency_median_s"),
        "last_third_latency_mean_s": mean("batch_last_third_latency_median_s"),
        "drift_abs_mean_s": mean("batch_latency_drift_abs_s"),
        "drift_pct_mean": mean("batch_latency_drift_pct"),
        "timeout_rate_mean": mean("timeout_rate"),
        "error_rate_mean": mean("error_rate"),
    }


def classify_drift(drift_pct_mean: float) -> str:
    if drift_pct_mean <= 0.05:
        return "stable"
    if drift_pct_mean <= 0.10:
        return "borderline"
    return "unstable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", help="Batch summary CSV files")
    args = parser.parse_args()

    paths = [Path(p) for p in args.summaries]
    report = aggregate(paths)
    if not report:
        print("No batch summaries provided.")
        return

    drift_class = classify_drift(report["drift_pct_mean"])

    print("\nCalibration Report:\n")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"drift_class: {drift_class}")

    if drift_class == "stable":
        print("recommendation: current batch size/pause policy is acceptable")
    elif drift_class == "borderline":
        print("recommendation: consider a longer pause or smaller batch size")
    else:
        print("recommendation: reduce batch size and/or increase pause time before final runs")


if __name__ == "__main__":
    main()
