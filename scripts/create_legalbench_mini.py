import argparse
import copy
import json
import random
from pathlib import Path

from src.config.defaults import BENCHMARKS_DIR, LEGALBENCH_MINI_PATH


BENCHMARK_ORDER = ("privacy_qa", "contractnli", "maud", "cuad")
MAX_TESTS_PER_BENCHMARK = 194
SORT_BY_DOCUMENT = True


def _select_tests(benchmark_name: str, tests: list[dict]) -> list[dict]:
    selected = [copy.deepcopy(test) for test in tests]
    if len(selected) > MAX_TESTS_PER_BENCHMARK:
        if SORT_BY_DOCUMENT:
            selected = sorted(
                selected,
                key=lambda test: (
                    random.seed((test.get("snippets") or [{}])[0].get("file_path", "")),
                    random.random(),
                )[1],
            )
        else:
            random.seed(benchmark_name)
            random.shuffle(selected)
        selected = selected[:MAX_TESTS_PER_BENCHMARK]
    for test in selected:
        test["tags"] = [benchmark_name]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks-dir", type=Path, default=BENCHMARKS_DIR)
    parser.add_argument("--output-path", type=Path, default=LEGALBENCH_MINI_PATH)
    args = parser.parse_args()

    all_tests: list[dict] = []
    counts: dict[str, int] = {}
    for benchmark_name in BENCHMARK_ORDER:
        benchmark_path = args.benchmarks_dir / f"{benchmark_name}.json"
        data = json.loads(benchmark_path.read_text(encoding="utf-8"))
        tests = data.get("tests") or data.get("test_cases") or data
        selected = _select_tests(benchmark_name, tests)
        all_tests.extend(selected)
        counts[benchmark_name] = len(selected)

    payload = {
        "metadata": {
            "source": "Derived locally from official LegalBench-RAG benchmark.py mini sampling logic",
            "max_tests_per_benchmark": MAX_TESTS_PER_BENCHMARK,
            "sort_by_document": SORT_BY_DOCUMENT,
            "benchmark_order": list(BENCHMARK_ORDER),
            "counts": counts,
        },
        "tests": all_tests,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_tests)} tests to {args.output_path}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
