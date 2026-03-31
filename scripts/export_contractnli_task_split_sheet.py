import argparse
from pathlib import Path

from src.config.defaults import CONTRACTNLI_TASK_SPLIT_PATH
from src.data import load_legalbench_contractnli
from src.eval.task_split import load_contractnli_task_split, write_contractnli_task_split


def _preview(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(CONTRACTNLI_TASK_SPLIT_PATH))
    args = parser.parse_args()

    data = load_legalbench_contractnli()
    existing = load_contractnli_task_split(Path(args.output))
    cases = data.test_cases
    rows = []

    for idx, case in enumerate(cases):
        snippets = case.get("snippets") or []
        if not snippets:
            continue
        case_id = f"contractnli_test_{idx:04d}"
        file_paths = [snip.get("file_path", "") for snip in snippets if snip.get("file_path")]
        answers = [snip.get("answer", "") for snip in snippets if snip.get("answer")]
        rows.append(
            {
                "case_id": case_id,
                "task": existing.get(case_id, ""),
                "query": case.get("query", ""),
                "primary_file": file_paths[0] if file_paths else "",
                "gold_span_count": str(len(snippets)),
                "gold_evidence_preview": _preview(" || ".join(answers)),
                "gold_evidence_full": " || ".join(answers),
                "notes": "",
            }
        )

    write_contractnli_task_split(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
