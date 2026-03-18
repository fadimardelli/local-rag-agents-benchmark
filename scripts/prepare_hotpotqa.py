import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset

from src.config.defaults import HOTPOTQA_CASES_PATH, HOTPOTQA_CORPUS_PATH


def _iter_examples(split: str) -> Iterable[Dict]:
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    for row in ds:
        yield row


def _build_records(split: str) -> Tuple[List[Dict], List[Dict]]:
    corpus_map: Dict[str, str] = {}
    cases: List[Dict] = []

    for row in _iter_examples(split):
        question = row.get("question")
        answer = row.get("answer")
        context = row.get("context", {})
        supporting = row.get("supporting_facts", {})
        context_titles = context.get("title", [])
        context_sentences = context.get("sentences", [])
        supporting_titles = sorted(set(supporting.get("title", [])))

        if not question or not answer:
            continue

        for title, sentences in zip(context_titles, context_sentences):
            if title not in corpus_map:
                corpus_map[title] = " ".join(sentences)

        cases.append(
            {
                "id": row.get("id"),
                "question": question,
                "answer": answer,
                "supporting_titles": supporting_titles,
            }
        )

    corpus = [{"title": title, "text": text} for title, text in corpus_map.items()]
    return cases, corpus


def _write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--cases-path", default=str(HOTPOTQA_CASES_PATH))
    parser.add_argument("--corpus-path", default=str(HOTPOTQA_CORPUS_PATH))
    args = parser.parse_args()

    cases, corpus = _build_records(args.split)
    _write_jsonl(cases, Path(args.cases_path))
    _write_jsonl(corpus, Path(args.corpus_path))

    print(f"Cases written: {len(cases)} -> {args.cases_path}")
    print(f"Corpus docs written: {len(corpus)} -> {args.corpus_path}")


if __name__ == "__main__":
    main()
