import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def plot_latency(rows_a, rows_b, label_a, label_b, out_path: Path):
    lat_a = [r["latency_s"] for r in rows_a]
    lat_b = [r["latency_s"] for r in rows_b]
    data = [lat_a, lat_b]

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=[label_a, label_b], showfliers=True)
    plt.ylabel("Latency (s)")
    plt.title("Latency Distribution")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_metric(rows_a, rows_b, key, label_a, label_b, out_path: Path, title: str):
    val_a = sum(r[key] for r in rows_a) / len(rows_a)
    val_b = sum(r[key] for r in rows_b) / len(rows_b)

    plt.figure(figsize=(5, 4))
    plt.bar([label_a, label_b], [val_a, val_b])
    plt.ylabel(key)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Traditional JSONL results")
    parser.add_argument("--b", required=True, help="Agentic JSONL results")
    parser.add_argument("--out-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--label-a", default="Traditional")
    parser.add_argument("--label-b", default="Agentic")
    args = parser.parse_args()

    rows_a = load_jsonl(Path(args.a))
    rows_b = load_jsonl(Path(args.b))

    out_dir = Path(args.out_dir)
    plot_latency(rows_a, rows_b, args.label_a, args.label_b, out_dir / "latency_boxplot.png")
    if "evidence_recall" in rows_a[0]:
        plot_metric(
            rows_a,
            rows_b,
            "evidence_recall",
            args.label_a,
            args.label_b,
            out_dir / "evidence_recall_bar.png",
            "Evidence Recall (mean)",
        )
    if "correct" in rows_a[0]:
        vals_a = [1.0 if r["correct"] else 0.0 for r in rows_a if r["correct"] is not None]
        vals_b = [1.0 if r["correct"] else 0.0 for r in rows_b if r["correct"] is not None]
        if vals_a and vals_b:
            plt.figure(figsize=(5, 4))
            plt.bar([args.label_a, args.label_b], [sum(vals_a) / len(vals_a), sum(vals_b) / len(vals_b)])
            plt.ylabel("accuracy")
            plt.title("Answer Accuracy (mean)")
            plt.tight_layout()
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / "accuracy_bar.png")
            plt.close()


if __name__ == "__main__":
    main()
