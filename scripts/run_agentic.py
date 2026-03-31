import argparse
from pathlib import Path

from src.rag import AgenticRAG, MultiActionAgenticRAG
from src.rag.llama_cpp import LlamaCppModel
from src.rag.retriever import DATASET_CHOICES, Retriever, resolve_dataset_paths
from src.utils.memory import PeakMemory
from src.utils.timers import Timer
from src.config.defaults import (
    LLAMA_GGUF_PATH_3B,
    LLAMA_GGUF_PATH_8B,
    LLAMA_MAX_TOKENS,
    LLAMA_N_CTX,
    LLAMA_N_GPU_LAYERS,
    LLAMA_SEED,
    LLAMA_TEMPERATURE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask the agentic RAG system")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="legalbench_contractnli")
    parser.add_argument("--model", choices=["8b", "3b"], default="8b")
    parser.add_argument("--workflow", choices=["legacy", "multi_action"], default="multi_action")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=LLAMA_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=LLAMA_MAX_TOKENS)
    args = parser.parse_args()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = LLAMA_GGUF_PATH_3B if args.model == "3b" else LLAMA_GGUF_PATH_8B

    index_path, meta_path = resolve_dataset_paths(args.dataset)
    model = LlamaCppModel(
        model_path=model_path,
        n_ctx=LLAMA_N_CTX,
        n_gpu_layers=LLAMA_N_GPU_LAYERS,
        seed=LLAMA_SEED,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    retriever = Retriever(index_path=index_path, meta_path=meta_path)
    rag = (
        MultiActionAgenticRAG(model=model, retriever=retriever)
        if args.workflow == "multi_action"
        else AgenticRAG(model=model, retriever=retriever)
    )
    with PeakMemory() as mem, Timer() as timer:
        result = rag.run(args.question)

    print("\nAnswer:\n")
    print(result.answer)
    print(f"\nDataset: {args.dataset}")
    print(f"\nLatency (s): {timer.elapsed:.2f}")
    print(f"Peak RSS (MB): {mem.peak_rss_bytes / (1024 * 1024):.1f}")
    print(f"Avg CPU (%): {mem.avg_cpu_percent:.1f}")
    print(f"\nController iterations: {result.iterations}")
    if hasattr(result, "retrieval_steps"):
        print(f"Retrieval steps: {getattr(result, 'retrieval_steps')}")
    if result.refined_queries:
        print("\nRefined queries:")
        for i, q in enumerate(result.refined_queries, 1):
            print(f"[{i}] {q}")
    if hasattr(result, "action_history") and result.action_history:
        print("\nRetrieval action history:")
        for i, action in enumerate(result.action_history, 1):
            print(f"[{i}] {action}")

    print("\nRetrieved chunks:\n")
    for i, chunk in enumerate(result.used, 1):
        print(f"[{i}] {chunk.doc_path} ({chunk.start}-{chunk.end}) score={chunk.score:.4f}")
        print(chunk.text.strip())
        print("-" * 80)


if __name__ == "__main__":
    main()
