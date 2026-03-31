import argparse
import json
import statistics as st
from pathlib import Path

from src.eval.datasets import iter_contractnli_cases_window
from src.eval.metrics import evidence_recall_multi_file
from src.rag.retriever import RetrievedChunk


def chunk_from_ref(ref: dict) -> RetrievedChunk:
    return RetrievedChunk(
        doc_path=ref["doc_path"],
        start=int(ref["start"]),
        end=int(ref["end"]),
        text="",
        score=float(ref.get("score", 0.0)),
    )


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Traced rerun JSONL from run_eval_contractnli.py")
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--baseline-jsonl", default="", help="Optional original batch JSONL for side-by-side latency compare")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.jsonl))
    cases = list(iter_contractnli_cases_window(offset=args.offset, limit=args.limit))
    if len(rows) != len(cases):
        raise ValueError(f"Result rows ({len(rows)}) do not match cases ({len(cases)})")

    total_queries = len(rows)
    improved_any = 0
    improved_to_final = 0
    worsened_vs_first = 0
    unchanged_all = 0
    hit_progressions = []
    ev_progressions = []
    extra_iter_queries = 0

    for row, case in zip(rows, cases):
        trace = row.get("iteration_trace") or []
        if not trace:
            continue
        hit_seq = []
        ev_seq = []
        for step in trace:
            retrieved = [chunk_from_ref(ref) for ref in step.get("retrieved_refs", [])]
            used = [chunk_from_ref(ref) for ref in step.get("used_refs", [])]
            hit_seq.append(1 if evidence_recall_multi_file(retrieved, case.gold_spans) else 0)
            ev_seq.append(1 if evidence_recall_multi_file(used, case.gold_spans) else 0)
        hit_progressions.append(hit_seq)
        ev_progressions.append(ev_seq)
        if len(hit_seq) > 1:
            extra_iter_queries += 1
        if any(v > hit_seq[0] for v in hit_seq[1:]) or any(v > ev_seq[0] for v in ev_seq[1:]):
            improved_any += 1
        if hit_seq[-1] > hit_seq[0] or ev_seq[-1] > ev_seq[0]:
            improved_to_final += 1
        if hit_seq[-1] < hit_seq[0] or ev_seq[-1] < ev_seq[0]:
            worsened_vs_first += 1
        if len(set(hit_seq)) == 1 and len(set(ev_seq)) == 1:
            unchanged_all += 1

    print("Trace Analysis\n")
    print(f"queries: {total_queries}")
    print(f"queries_with_extra_iterations: {extra_iter_queries}")
    print(f"queries_with_any_retrieval_improvement_after_iteration1: {improved_any}")
    print(f"queries_with_final_retrieval_better_than_iteration1: {improved_to_final}")
    print(f"queries_with_final_retrieval_worse_than_iteration1: {worsened_vs_first}")
    print(f"queries_with_no_retrieval_change_across_iterations: {unchanged_all}")

    if hit_progressions:
        max_steps = max(len(s) for s in hit_progressions)
        print("\nStep-Level Retrieval")
        for step_idx in range(max_steps):
            step_hits = [seq[step_idx] for seq in hit_progressions if len(seq) > step_idx]
            step_evs = [seq[step_idx] for seq in ev_progressions if len(seq) > step_idx]
            print(
                f"iteration_{step_idx + 1}_topk_hit_rate: {sum(step_hits) / len(step_hits):.4f} "
                f"(n={len(step_hits)})"
            )
            print(
                f"iteration_{step_idx + 1}_evidence_recall: {sum(step_evs) / len(step_evs):.4f} "
                f"(n={len(step_evs)})"
            )

    if args.baseline_jsonl:
        baseline_rows = load_jsonl(Path(args.baseline_jsonl))
        if len(baseline_rows) != len(rows):
            raise ValueError("Baseline JSONL row count does not match traced rerun")
        deltas = [row["latency_s"] - base["latency_s"] for row, base in zip(rows, baseline_rows)]
        print("\nBaseline Comparison")
        print(f"latency_delta_mean_s: {st.mean(deltas):.4f}")
        print(f"latency_delta_median_s: {st.median(deltas):.4f}")
        faster = sum(1 for d in deltas if d < 0)
        slower = sum(1 for d in deltas if d > 0)
        same = len(deltas) - faster - slower
        print(f"queries_faster_than_baseline: {faster}")
        print(f"queries_slower_than_baseline: {slower}")
        print(f"queries_same_as_baseline: {same}")


if __name__ == "__main__":
    main()
