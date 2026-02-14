import argparse
from pathlib import Path

from src.rag import AgenticRAG
from src.config.defaults import LLAMA_GGUF_PATH_3B, LLAMA_GGUF_PATH_8B


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask the agentic RAG system")
    parser.add_argument("--model", choices=["8b", "3b"], default="8b")
    parser.add_argument("--model-path", type=str, default="")
    args = parser.parse_args()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = LLAMA_GGUF_PATH_3B if args.model == "3b" else LLAMA_GGUF_PATH_8B

    rag = AgenticRAG(model_path=model_path)
    result = rag.run(args.question)

    print("\nAnswer:\n")
    print(result.answer)
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
