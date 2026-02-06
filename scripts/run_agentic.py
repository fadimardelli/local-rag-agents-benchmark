import argparse

from src.rag import AgenticRAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask the agentic RAG system")
    args = parser.parse_args()

    rag = AgenticRAG()
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
