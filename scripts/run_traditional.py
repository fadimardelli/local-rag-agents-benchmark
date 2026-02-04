import argparse

from src.rag import TraditionalRAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask the RAG system")
    args = parser.parse_args()

    rag = TraditionalRAG()
    result = rag.run(args.question)

    print("\nAnswer:\n")
    print(result.answer)
    print("\nRetrieved chunks:\n")
    for i, chunk in enumerate(result.used, 1):
        print(f"[{i}] {chunk.doc_path} ({chunk.start}-{chunk.end}) score={chunk.score:.4f}")
        print(chunk.text.strip())
        print("-" * 80)


if __name__ == "__main__":
    main()
