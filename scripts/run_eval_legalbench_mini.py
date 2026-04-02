import argparse
from pathlib import Path

from src.eval.datasets import iter_legalbench_mini_cases_window
from src.eval.harness import run_eval, select_summary_view, summarize, write_results
from src.config.defaults import (
    HYBRID_BM25_TOP_K,
    HYBRID_DENSE_TOP_K,
    HYBRID_RRF_K,
    LLAMA_GGUF_PATH_3B,
    LLAMA_GGUF_PATH_8B,
    RERANK_CANDIDATE_K,
    RERANK_ENABLED,
    RERANK_MODEL_NAME,
    QUERY_TRANSFORM_MODE,
    RETRIEVAL_MODE,
)


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
    parser.add_argument("--query-transform-mode", choices=["none", "hyde"], default=QUERY_TRANSFORM_MODE)
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default=RETRIEVAL_MODE)
    parser.add_argument("--hybrid-dense-top-k", type=int, default=HYBRID_DENSE_TOP_K)
    parser.add_argument("--hybrid-bm25-top-k", type=int, default=HYBRID_BM25_TOP_K)
    parser.add_argument("--hybrid-rrf-k", type=int, default=HYBRID_RRF_K)
    parser.add_argument("--rerank-enabled", action="store_true", default=RERANK_ENABLED)
    parser.add_argument("--rerank-candidate-k", type=int, default=RERANK_CANDIDATE_K)
    parser.add_argument("--rerank-model-name", type=str, default=RERANK_MODEL_NAME)
    parser.add_argument(
        "--trace-agentic",
        action="store_true",
        help="Include per-iteration agentic trace data in JSONL/CSV outputs",
    )
    args = parser.parse_args()

    cases = list(iter_legalbench_mini_cases_window(offset=args.offset, limit=args.limit))
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
