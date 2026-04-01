import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, List

from src.config.defaults import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTRACTNLI_ORIG_INDEX_PATH,
    CONTRACTNLI_ORIG_META_PATH,
    EMBEDDING_MODEL_NAME,
    HYBRID_BM25_TOP_K,
    HYBRID_DENSE_TOP_K,
    HYBRID_RRF_K,
    HOTPOTQA_INDEX_PATH,
    HOTPOTQA_META_PATH,
    LLAMA_GGUF_PATH_3B,
    LLAMA_GGUF_PATH_8B,
    QUERY_TRANSFORM_MODE,
    RERANK_CANDIDATE_K,
    RERANK_ENABLED,
    RETRIEVAL_MODE,
)
from src.eval.datasets import (
    ContractNLIOriginalCase,
    ContractNLITestCase,
    HotpotQACase,
    iter_contractnli_cases_window,
    iter_contractnli_original_cases,
    iter_hotpotqa_cases,
)
from src.eval.metrics import (
    answer_correctness,
    answer_exact_match,
    answer_f1,
    evidence_recall_multi,
    evidence_recall_multi_file,
    supporting_doc_recall,
)
from src.rag import Retriever, TraditionalRAG
from src.utils.memory import PeakMemory
from src.utils.timers import Timer


@dataclass
class RetrievalSelectionRow:
    config_name: str
    dataset: str
    case_key: str
    query: str
    retrieval_query: str
    answer: str
    topk_hit: bool
    evidence_recall: bool
    supporting_doc_recall: float | None
    answer_accuracy: float | None
    answer_f1: float | None
    retrieval_eval_applicable: bool
    latency_s: float
    retrieval_latency_s: float
    generation_latency_s: float
    peak_rss_bytes: int
    structure_json: str


def _load_legalbench_pool(path: Path) -> List[tuple[ContractNLITestCase, dict]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            offset = int(row["offset"])
            case = next(iter_contractnli_cases_window(offset=offset, limit=1))
            rows.append((case, row))
    return rows


def _load_contractnli_original_pool(path: Path) -> List[tuple[ContractNLIOriginalCase, dict]]:
    by_key = {}
    for case in iter_contractnli_original_cases():
        by_key[(case.query, case.file_name)] = case

    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["query"], row["file_name"])
            case = by_key[key]
            rows.append((case, row))
    return rows


def _load_hotpot_pool(path: Path) -> List[tuple[HotpotQACase, dict]]:
    by_offset = {}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        offset = int(row["offset"])
        if offset not in by_offset:
            by_offset[offset] = next(iter_hotpotqa_cases(offset=offset, limit=1))
    return [(by_offset[int(row["offset"])], row) for row in rows]


def _dataset_config_summary(top_k: int, retrieval_mode: str, hybrid_dense_top_k: int, hybrid_bm25_top_k: int, hybrid_rrf_k: int) -> dict:
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_mode": retrieval_mode,
        "rerank_enabled": args.rerank_enabled,
        "rerank_candidate_k": args.rerank_candidate_k,
        "query_transform_mode": QUERY_TRANSFORM_MODE,
        "hybrid_dense_top_k": hybrid_dense_top_k,
        "hybrid_bm25_top_k": hybrid_bm25_top_k,
        "hybrid_rrf_k": hybrid_rrf_k,
        "top_k": top_k,
    }


def _summarize_dataset(rows: Iterable[RetrievalSelectionRow]) -> dict:
    rows = list(rows)
    if not rows:
        return {}
    retrieval_rows = [r for r in rows if r.retrieval_eval_applicable]
    summary = {
        "count": len(rows),
        "retrieval_eval_count": len(retrieval_rows),
        "latency_median_s": sorted(r.latency_s for r in rows)[len(rows) // 2],
        "latency_mean_s": mean(r.latency_s for r in rows),
        "retrieval_latency_mean_s": mean(r.retrieval_latency_s for r in rows),
        "generation_latency_mean_s": mean(r.generation_latency_s for r in rows),
        "peak_rss_bytes_max": max(r.peak_rss_bytes for r in rows),
    }
    if retrieval_rows:
        summary["topk_hit_rate"] = mean(1.0 if r.topk_hit else 0.0 for r in retrieval_rows)
        summary["evidence_recall_mean"] = mean(1.0 if r.evidence_recall else 0.0 for r in retrieval_rows)
    answer_vals = [r.answer_accuracy for r in rows if r.answer_accuracy is not None]
    if answer_vals:
        summary["answer_accuracy"] = mean(answer_vals)
    support_vals = [r.supporting_doc_recall for r in rows if r.supporting_doc_recall is not None]
    if support_vals:
        summary["supporting_doc_recall_mean"] = mean(support_vals)
    f1_vals = [r.answer_f1 for r in rows if r.answer_f1 is not None]
    if f1_vals:
        summary["answer_f1_mean"] = mean(f1_vals)
    return summary


def _macro_summary(grouped: dict[str, List[RetrievalSelectionRow]]) -> dict:
    dataset_summaries = {name: _summarize_dataset(rows) for name, rows in grouped.items()}
    macro = {
        "macro_latency_mean_s": mean(v["latency_mean_s"] for v in dataset_summaries.values()),
    }
    retrieval_summaries = [v for v in dataset_summaries.values() if "topk_hit_rate" in v]
    if retrieval_summaries:
        macro["macro_topk_hit_rate"] = mean(v["topk_hit_rate"] for v in retrieval_summaries)
        macro["macro_evidence_recall_mean"] = mean(v["evidence_recall_mean"] for v in retrieval_summaries)
    if all("answer_accuracy" in v for v in dataset_summaries.values()):
        macro["macro_answer_accuracy"] = mean(v["answer_accuracy"] for v in dataset_summaries.values())
    if "hotpotqa" in dataset_summaries and "supporting_doc_recall_mean" in dataset_summaries["hotpotqa"]:
        macro["hotpot_supporting_doc_recall_mean"] = dataset_summaries["hotpotqa"]["supporting_doc_recall_mean"]
    return macro


def _run_legalbench(
    rag: TraditionalRAG,
    cases_with_meta: List[tuple[ContractNLITestCase, dict]],
    config_name: str,
) -> List[RetrievalSelectionRow]:
    out = []
    for case, meta in cases_with_meta:
        with PeakMemory() as mem, Timer() as timer:
            result = rag.run(case.query, top_k=args.top_k)
        out.append(
            RetrievalSelectionRow(
                config_name=config_name,
                dataset="legalbench_contractnli",
                case_key=case.case_id,
                query=case.query,
                retrieval_query=result.retrieval_query,
                answer=result.answer,
                topk_hit=evidence_recall_multi_file(result.retrieved, case.gold_spans),
                evidence_recall=evidence_recall_multi_file(result.used, case.gold_spans),
                supporting_doc_recall=None,
                answer_accuracy=None,
                answer_f1=None,
                retrieval_eval_applicable=True,
                latency_s=timer.elapsed,
                retrieval_latency_s=result.retrieval_latency_s,
                generation_latency_s=result.generation_latency_s,
                peak_rss_bytes=mem.peak_rss_bytes,
                structure_json=json.dumps(meta, ensure_ascii=False),
            )
        )
    return out


def _run_contractnli_original(
    rag: TraditionalRAG,
    cases_with_meta: List[tuple[ContractNLIOriginalCase, dict]],
    config_name: str,
) -> List[RetrievalSelectionRow]:
    out = []
    for case, meta in cases_with_meta:
        with PeakMemory() as mem, Timer() as timer:
            result = rag.run_label(case.query, top_k=args.top_k)
        retrieval_eval_applicable = len(case.gold_spans) > 0
        out.append(
            RetrievalSelectionRow(
                config_name=config_name,
                dataset="contractnli_original",
                case_key=f"{case.file_name}::{case.query}",
                query=case.query,
                retrieval_query=result.retrieval_query,
                answer=result.answer,
                topk_hit=evidence_recall_multi(result.retrieved, case.file_name, case.gold_spans),
                evidence_recall=evidence_recall_multi(result.used, case.file_name, case.gold_spans),
                supporting_doc_recall=None,
                answer_accuracy=1.0 if answer_correctness(result.answer, case.gold_label) else 0.0,
                answer_f1=None,
                retrieval_eval_applicable=retrieval_eval_applicable,
                latency_s=timer.elapsed,
                retrieval_latency_s=result.retrieval_latency_s,
                generation_latency_s=result.generation_latency_s,
                peak_rss_bytes=mem.peak_rss_bytes,
                structure_json=json.dumps(meta, ensure_ascii=False),
            )
        )
    return out


def _run_hotpot(
    rag: TraditionalRAG,
    cases_with_meta: List[tuple[HotpotQACase, dict]],
    config_name: str,
) -> List[RetrievalSelectionRow]:
    out = []
    for case, meta in cases_with_meta:
        with PeakMemory() as mem, Timer() as timer:
            result = rag.run_hotpot(case.query, top_k=args.top_k)
        retrieved_support_recall = supporting_doc_recall(result.retrieved, case.supporting_titles)
        supp_recall = supporting_doc_recall(result.used, case.supporting_titles)
        out.append(
            RetrievalSelectionRow(
                config_name=config_name,
                dataset="hotpotqa",
                case_key=meta["id"],
                query=case.query,
                retrieval_query=result.retrieval_query,
                answer=result.answer,
                topk_hit=retrieved_support_recall == 1.0,
                evidence_recall=supp_recall == 1.0,
                supporting_doc_recall=supp_recall,
                answer_accuracy=1.0 if answer_exact_match(result.answer, case.gold_answer) else 0.0,
                answer_f1=answer_f1(result.answer, case.gold_answer),
                retrieval_eval_applicable=True,
                latency_s=timer.elapsed,
                retrieval_latency_s=result.retrieval_latency_s,
                generation_latency_s=result.generation_latency_s,
                peak_rss_bytes=mem.peak_rss_bytes,
                structure_json=json.dumps(meta, ensure_ascii=False),
            )
        )
    return out


def _write_jsonl(rows: List[RetrievalSelectionRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _write_csv(rows: List[RetrievalSelectionRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_markdown_report(
    *,
    path: Path,
    config_name: str,
    config_summary: dict,
    grouped: dict[str, List[RetrievalSelectionRow]],
    dataset_summaries: dict[str, dict],
    macro_summary: dict,
) -> None:
    def fmt_optional(summary: dict, key: str) -> str:
        value = summary.get(key)
        if value is None:
            return "N/A"
        return f"{value:.3f}"

    lines = []
    lines.append(f"# Retrieval Selection Report: {config_name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Active Retrieval Config")
    lines.append("")
    for key, value in config_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append("| Dataset | Count | Top-k Hit | Evidence Recall | Answer Accuracy | Supporting Doc Recall | Answer F1 | Latency Median (s) | Retrieval Mean (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    ordered_datasets = [
        name
        for name in ("legalbench_contractnli", "contractnli_original", "hotpotqa")
        if name in dataset_summaries
    ]
    for dataset in ordered_datasets:
        summary = dataset_summaries[dataset]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    f'{summary["count"]} (retrieval n={summary["retrieval_eval_count"]})',
                    fmt_optional(summary, "topk_hit_rate"),
                    fmt_optional(summary, "evidence_recall_mean"),
                    fmt_optional(summary, "answer_accuracy"),
                    fmt_optional(summary, "supporting_doc_recall_mean"),
                    fmt_optional(summary, "answer_f1_mean"),
                    f'{summary["latency_median_s"]:.3f}',
                    f'{summary["retrieval_latency_mean_s"]:.3f}',
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Macro Summary")
    lines.append("")
    for key, value in macro_summary.items():
        lines.append(f"- `{key}`: `{value:.3f}`")
    lines.append("")
    lines.append("## Next Use")
    lines.append("")
    lines.append("- Compare this report against other retrieval configs using the same candidate pools.")
    lines.append("- After baseline runs, use these per-case rows to label `easy`, `medium`, and `hard` retrieval difficulty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--config-name", required=True, help="Short label for this retrieval setup")
parser.add_argument("--output-dir", required=True, help="Directory where JSONL/CSV/MD summary will be written")
parser.add_argument("--top-k", type=int, default=5)
parser.add_argument("--model", choices=["3b", "8b"], default="3b")
parser.add_argument("--warmup", action="store_true")
parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default=RETRIEVAL_MODE)
parser.add_argument("--hybrid-dense-top-k", type=int, default=HYBRID_DENSE_TOP_K)
parser.add_argument("--hybrid-bm25-top-k", type=int, default=HYBRID_BM25_TOP_K)
parser.add_argument("--hybrid-rrf-k", type=int, default=HYBRID_RRF_K)
parser.add_argument("--rerank-enabled", action="store_true", default=RERANK_ENABLED)
parser.add_argument("--rerank-candidate-k", type=int, default=RERANK_CANDIDATE_K)
parser.add_argument(
    "--datasets",
    nargs="+",
    choices=["legalbench_contractnli", "contractnli_original", "hotpotqa"],
    default=["legalbench_contractnli", "contractnli_original", "hotpotqa"],
    help="Subset of candidate-pool datasets to evaluate",
)
args = parser.parse_args()

model_path = LLAMA_GGUF_PATH_3B if args.model == "3b" else LLAMA_GGUF_PATH_8B
output_dir = Path(args.output_dir)

selected = set(args.datasets)

legalbench_cases = None
original_cases = None
hotpot_cases = None

legalbench_rag = None
original_rag = None
hotpot_rag = None

if "legalbench_contractnli" in selected:
    legalbench_cases = _load_legalbench_pool(
        Path("data/annotations/retrieval_selection/legalbench_contractnli_candidate_pool.csv")
    )
    legalbench_rag = TraditionalRAG(
        retriever=Retriever(
            retrieval_mode=args.retrieval_mode,
            hybrid_dense_top_k=args.hybrid_dense_top_k,
            hybrid_bm25_top_k=args.hybrid_bm25_top_k,
            hybrid_rrf_k=args.hybrid_rrf_k,
            rerank_enabled=args.rerank_enabled,
            rerank_candidate_k=args.rerank_candidate_k,
        ),
        model_path=model_path,
    )

if "contractnli_original" in selected:
    original_cases = _load_contractnli_original_pool(
        Path("data/annotations/retrieval_selection/contractnli_original_candidate_pool.csv")
    )
    original_rag = TraditionalRAG(
        retriever=Retriever(
            index_path=CONTRACTNLI_ORIG_INDEX_PATH,
            meta_path=CONTRACTNLI_ORIG_META_PATH,
            retrieval_mode=args.retrieval_mode,
            hybrid_dense_top_k=args.hybrid_dense_top_k,
            hybrid_bm25_top_k=args.hybrid_bm25_top_k,
            hybrid_rrf_k=args.hybrid_rrf_k,
            rerank_enabled=args.rerank_enabled,
            rerank_candidate_k=args.rerank_candidate_k,
        ),
        model_path=model_path,
    )

if "hotpotqa" in selected:
    hotpot_cases = _load_hotpot_pool(
        Path("data/annotations/retrieval_selection/hotpotqa_candidate_pool.csv")
    )
    hotpot_rag = TraditionalRAG(
        retriever=Retriever(
            index_path=HOTPOTQA_INDEX_PATH,
            meta_path=HOTPOTQA_META_PATH,
            retrieval_mode=args.retrieval_mode,
            hybrid_dense_top_k=args.hybrid_dense_top_k,
            hybrid_bm25_top_k=args.hybrid_bm25_top_k,
            hybrid_rrf_k=args.hybrid_rrf_k,
            rerank_enabled=args.rerank_enabled,
            rerank_candidate_k=args.rerank_candidate_k,
        ),
        model_path=model_path,
    )

if args.warmup:
    if legalbench_rag is not None:
        legalbench_rag.run(legalbench_cases[0][0].query, top_k=args.top_k)
    if original_rag is not None:
        original_rag.run_label(original_cases[0][0].query, top_k=args.top_k)
    if hotpot_rag is not None:
        hotpot_rag.run_hotpot(hotpot_cases[0][0].query, top_k=args.top_k)

all_rows = []
if legalbench_rag is not None:
    all_rows.extend(_run_legalbench(legalbench_rag, legalbench_cases, args.config_name))
if original_rag is not None:
    all_rows.extend(_run_contractnli_original(original_rag, original_cases, args.config_name))
if hotpot_rag is not None:
    all_rows.extend(_run_hotpot(hotpot_rag, hotpot_cases, args.config_name))

grouped = {name: [r for r in all_rows if r.dataset == name] for name in args.datasets}
dataset_summaries = {name: _summarize_dataset(rows) for name, rows in grouped.items()}
macro_summary = _macro_summary(grouped)
config_summary = _dataset_config_summary(
    args.top_k,
    args.retrieval_mode,
    args.hybrid_dense_top_k,
    args.hybrid_bm25_top_k,
    args.hybrid_rrf_k,
)

_write_jsonl(all_rows, output_dir / "results.jsonl")
_write_csv(all_rows, output_dir / "results.csv")
(output_dir / "summary.json").write_text(
    json.dumps(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_name": args.config_name,
            "config": config_summary,
            "datasets": dataset_summaries,
            "macro": macro_summary,
        },
        indent=2,
    ),
    encoding="utf-8",
)
_write_markdown_report(
    path=output_dir / "report.md",
    config_name=args.config_name,
    config_summary=config_summary,
    grouped=grouped,
    dataset_summaries=dataset_summaries,
    macro_summary=macro_summary,
)

print("\nRetrieval Selection Summary\n")
for dataset, summary in dataset_summaries.items():
    print(dataset)
    for key in (
        "count",
        "retrieval_eval_count",
        "topk_hit_rate",
        "evidence_recall_mean",
        "answer_accuracy",
        "supporting_doc_recall_mean",
        "answer_f1_mean",
        "latency_median_s",
    ):
        if key in summary:
            print(f"  {key}: {summary[key]}")
print("\nMacro")
for key, value in macro_summary.items():
    print(f"  {key}: {value}")
print(f"\nOutputs written to {output_dir}")

for rag in (legalbench_rag, original_rag, hotpot_rag):
    if rag is not None:
        rag.model.close()
