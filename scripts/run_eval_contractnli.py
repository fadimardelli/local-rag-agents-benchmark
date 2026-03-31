import argparse
from pathlib import Path

from src.eval.datasets import iter_contractnli_cases_window
from src.eval.harness import run_eval, select_summary_view, summarize, write_results
from src.config.defaults import LLAMA_GGUF_PATH_3B, LLAMA_GGUF_PATH_8B


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["traditional", "agentic"], required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-jsonl", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--summary-view", choices=["all", "thesis", "calibration"], default="thesis")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--batch-id", type=str, default="")
    parser.add_argument("--warmup", action="store_true", help="Run one warm-up query before timing")
    parser.add_argument("--model", choices=["8b", "3b"], default="8b")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--trace-agentic",
        action="store_true",
        help="Include per-iteration agentic trace data in JSONL/CSV outputs",
    )
    args = parser.parse_args()

    cases = list(iter_contractnli_cases_window(offset=args.offset, limit=args.limit))
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = LLAMA_GGUF_PATH_3B if args.model == "3b" else LLAMA_GGUF_PATH_8B

    batch_id = args.batch_id or f"off{args.offset}_lim{args.limit}"
    results = run_eval(
        cases,
        mode=args.mode,
        warmup=args.warmup,
        model_path=model_path,
        run_id=args.run_id,
        batch_id=batch_id,
        trace_agentic=args.trace_agentic,
        top_k=args.top_k,
    )

    summary = select_summary_view(summarize(results), args.summary_view)
    print("\nSummary:\n")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.output_jsonl:
        write_results(results, Path(args.output_jsonl))
        print(f"\nResults written to {args.output_jsonl}")
    if args.output_csv:
        from src.eval.harness import write_results_csv

        write_results_csv(results, Path(args.output_csv))
        print(f"\nResults CSV written to {args.output_csv}")
    if args.summary_csv:
        from src.eval.harness import write_summary_csv

        write_summary_csv(summary, Path(args.summary_csv))
        print(f"\nSummary CSV written to {args.summary_csv}")


if __name__ == "__main__":
    main()
