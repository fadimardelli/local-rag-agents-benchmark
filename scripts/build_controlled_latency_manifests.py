import argparse
import json
import random
from pathlib import Path

from src.config.defaults import (
    HOTPOTQA_CASES_PATH,
    LEGALBENCH_MINI_BALANCED_PATH,
)
from src.data import load_legalbench_mini_balanced


DEFAULT_HOTPOT_COUNT = 100
DEFAULT_HOTPOT_POPULATION = 1000
DEFAULT_SEED = 42


def _legal_manifest() -> dict:
    data = load_legalbench_mini_balanced()
    cases = []
    for idx, case in enumerate(data.test_cases):
        tags = case.get("tags") or []
        cases.append(
            {
                "manifest_index": idx,
                "dataset_offset": idx,
                "query": case.get("query", ""),
                "subset": tags[0] if tags else "",
            }
        )

    return {
        "benchmark": "legalbench_mini_balanced",
        "selection_method": "fixed balanced subset",
        "source_path": str(LEGALBENCH_MINI_BALANCED_PATH),
        "count": len(cases),
        "cases": cases,
    }


def _hotpot_manifest(count: int, population_limit: int, seed: int) -> dict:
    if count > population_limit:
        raise ValueError("count must be <= population_limit")

    chosen_offsets = random.Random(seed).sample(range(population_limit), count)
    cases = []
    chosen_set = set(chosen_offsets)
    with HOTPOTQA_CASES_PATH.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= population_limit:
                break
            if idx not in chosen_set:
                continue
            row = json.loads(line)
            cases.append(
                {
                    "manifest_index": len(cases),
                    "dataset_offset": idx,
                    "query": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "supporting_titles": row.get("supporting_titles") or [],
                }
            )

    case_by_offset = {case["dataset_offset"]: case for case in cases}
    ordered_cases = [case_by_offset[offset] for offset in chosen_offsets]
    for manifest_index, case in enumerate(ordered_cases):
        case["manifest_index"] = manifest_index

    return {
        "benchmark": "hotpotqa",
        "selection_method": "fixed random sample from first full-benchmark slice",
        "source_path": str(HOTPOTQA_CASES_PATH),
        "population_limit": population_limit,
        "seed": seed,
        "count": len(ordered_cases),
        "cases": ordered_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/fadimardelli/local-rag-agents-benchmark/data/benchmarks/latency"),
    )
    parser.add_argument("--hotpot-count", type=int, default=DEFAULT_HOTPOT_COUNT)
    parser.add_argument("--hotpot-population", type=int, default=DEFAULT_HOTPOT_POPULATION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    legal_path = args.output_dir / "legalbench_mini_balanced_100_manifest.json"
    legal_payload = _legal_manifest()
    legal_path.write_text(json.dumps(legal_payload, indent=2), encoding="utf-8")

    hotpot_path = args.output_dir / "hotpotqa_first1000_sample100_manifest.json"
    hotpot_payload = _hotpot_manifest(
        count=args.hotpot_count,
        population_limit=args.hotpot_population,
        seed=args.seed,
    )
    hotpot_path.write_text(json.dumps(hotpot_payload, indent=2), encoding="utf-8")

    print(f"Wrote legal latency manifest to {legal_path}")
    print(f"Wrote Hotpot latency manifest to {hotpot_path}")


if __name__ == "__main__":
    main()
