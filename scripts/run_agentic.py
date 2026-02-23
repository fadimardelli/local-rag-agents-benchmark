import argparse
from pathlib import Path

from src.rag import AgenticRAG
from src.rag.llama_cpp import LlamaCppModel
from src.utils.memory import PeakMemory
from src.utils.timers import Timer
from src.config.defaults import (
    LLAMA_GGUF_PATH_3B,
    LLAMA_GGUF_PATH_8B,
    LLAMA_MAX_TOKENS,
    LLAMA_N_CTX,
    LLAMA_TEMPERATURE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask the agentic RAG system")
    parser.add_argument("--model", choices=["8b", "3b"], default="8b")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=LLAMA_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=LLAMA_MAX_TOKENS)
    args = parser.parse_args()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = LLAMA_GGUF_PATH_3B if args.model == "3b" else LLAMA_GGUF_PATH_8B

    model = LlamaCppModel(
        model_path=model_path,
        n_ctx=LLAMA_N_CTX,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    rag = AgenticRAG(model=model)
    with PeakMemory() as mem, Timer() as timer:
        result = rag.run(args.question)

    print("\nAnswer:\n")
    print(result.answer)
    print(f"\nLatency (s): {timer.elapsed:.2f}")
    print(f"Peak RSS (MB): {mem.peak_rss_bytes / (1024 * 1024):.1f}")
    print(f"Avg CPU (%): {mem.avg_cpu_percent:.1f}")
    print(f"\nIterations: {result.iterations}")
    if result.refined_queries:
        print("\nRefined queries:")
        for i, q in enumerate(result.refined_queries, 1):
            print(f"[{i}] {q}")

    print("\nRetrieved chunks:\n")
    for i, chunk in enumerate(result.used, 1):
        print(f"[{i}] {chunk.doc_path} ({chunk.start}-{chunk.end}) score={chunk.score:.4f}")
        print(chunk.text.strip())
        print("-" * 80)


if __name__ == "__main__":
    main()
