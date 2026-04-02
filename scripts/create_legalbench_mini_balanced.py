import argparse
import copy
import json
import random
from pathlib import Path

from src.config.defaults import LEGALBENCH_MINI_BALANCED_PATH, LEGALBENCH_MINI_PATH


DEFAULT_PER_BENCHMARK = 25
BENCHMARK_ORDER = ("privacy_qa", "contractnli", "maud", "cuad")
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=LEGALBENCH_MINI_PATH)
    parser.add_argument("--output-path", type=Path, default=LEGALBENCH_MINI_BALANCED_PATH)
    parser.add_argument("--per-benchmark", type=int, default=DEFAULT_PER_BENCHMARK)
    args = parser.parse_args()

    payload = json.loads(args.input_path.read_text(encoding="utf-8"))
    tests = payload.get("tests") or payload.get("test_cases") or payload

    grouped: dict[str, list[dict]] = {name: [] for name in BENCHMARK_ORDER}
    for test in tests:
        tags = test.get("tags") or []
        if not tags:
            continue
        tag = tags[0]
        if tag in grouped:
            grouped[tag].append(test)

    selected: list[dict] = []
    counts: dict[str, int] = {}
    for name in BENCHMARK_ORDER:
        items = [copy.deepcopy(item) for item in grouped[name]]
        rng = random.Random(f"{SEED}:{name}")
        rng.shuffle(items)
        chosen = items[: args.per_benchmark]
        selected.extend(chosen)
        counts[name] = len(chosen)

    combined_rng = random.Random(SEED)
    combined_rng.shuffle(selected)

    out = {
        "metadata": {
            "source": "Balanced sample derived locally from legalbenchrag_mini.json",
            "per_benchmark": args.per_benchmark,
            "seed": SEED,
            "counts": counts,
            "benchmark_order": list(BENCHMARK_ORDER),
        },
        "tests": selected,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} tests to {args.output_path}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
