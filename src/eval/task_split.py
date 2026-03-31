import csv
from pathlib import Path
from typing import Dict, List

from src.config.defaults import CONTRACTNLI_TASK_SPLIT_PATH

VALID_TASKS = {"", "T1", "T2"}


def load_contractnli_task_split(
    path: Path = CONTRACTNLI_TASK_SPLIT_PATH,
) -> Dict[str, str]:
    if not path.exists():
        return {}
    labels: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            task = (row.get("task") or "").strip()
            if not case_id:
                continue
            if task not in VALID_TASKS:
                raise ValueError(f"Invalid task label '{task}' for case_id '{case_id}'")
            labels[case_id] = task
    return labels


def write_contractnli_task_split(
    rows: List[Dict[str, str]],
    path: Path = CONTRACTNLI_TASK_SPLIT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "task",
        "query",
        "primary_file",
        "gold_span_count",
        "gold_evidence_preview",
        "gold_evidence_full",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_task_split(labels: Dict[str, str]) -> Dict[str, int]:
    summary = {"T1": 0, "T2": 0, "unlabeled": 0}
    for task in labels.values():
        if task == "T1":
            summary["T1"] += 1
        elif task == "T2":
            summary["T2"] += 1
        else:
            summary["unlabeled"] += 1
    return summary
