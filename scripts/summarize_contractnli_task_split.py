import argparse
from pathlib import Path

from src.config.defaults import CONTRACTNLI_TASK_SPLIT_PATH
from src.eval.task_split import load_contractnli_task_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(CONTRACTNLI_TASK_SPLIT_PATH))
    args = parser.parse_args()

    labels = load_contractnli_task_split(Path(args.input))
    total = len(labels)
    t1 = sum(1 for task in labels.values() if task == "T1")
    t2 = sum(1 for task in labels.values() if task == "T2")
    unlabeled = sum(1 for task in labels.values() if task == "")

    print("ContractNLI Task Split Summary:\n")
    print(f"total_rows: {total}")
    print(f"T1: {t1}")
    print(f"T2: {t2}")
    print(f"unlabeled: {unlabeled}")


if __name__ == "__main__":
    main()
