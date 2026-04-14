from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "thesis" / "submission_pack" / "full_results"
CSV_DIR = OUT_ROOT / "csv"
WORKBOOK_PATH = OUT_ROOT / "benchmark_full_results_workbook.xml"
ZIP_PATH = PROJECT_ROOT / "backups" / "full_results_submission_pack.zip"


@dataclass(frozen=True)
class RunSpec:
    key: str
    benchmark: str
    mode: str
    model: str
    scope: str
    label: str
    results_glob: str
    csv_name: str
    sheet_name: str
    notes: str = ""


RUN_SPECS = [
    RunSpec(
        key="legal_trad_llama_3b",
        benchmark="LegalBench-mini",
        mode="traditional",
        model="Llama 3B",
        scope="full_776",
        label="Traditional Llama 3B",
        results_glob="data/results/legalbench_mini/final/overnight_legalbench_mini_traditional_3b_off*.jsonl",
        csv_name="legalbench_mini_traditional_llama_3b_full_results.csv",
        sheet_name="legal_t3b",
    ),
    RunSpec(
        key="legal_agentic_llama_3b",
        benchmark="LegalBench-mini",
        mode="agentic",
        model="Llama 3B",
        scope="full_776",
        label="Agentic Llama 3B",
        results_glob="data/results/legalbench_mini/final/legalbench_mini_agentic_llama_3b_off*.jsonl",
        csv_name="legalbench_mini_agentic_llama_3b_full_results.csv",
        sheet_name="legal_a3b",
        notes="Extra model-choice run.",
    ),
    RunSpec(
        key="legal_trad_llama_8b",
        benchmark="LegalBench-mini",
        mode="traditional",
        model="Llama 8B",
        scope="full_776",
        label="Traditional Llama 8B",
        results_glob="data/results/legalbench_mini/final/overnight_legalbench_mini_traditional_8b_off*.jsonl",
        csv_name="legalbench_mini_traditional_llama_8b_full_results.csv",
        sheet_name="legal_t8b",
    ),
    RunSpec(
        key="legal_agentic_llama_8b",
        benchmark="LegalBench-mini",
        mode="agentic",
        model="Llama 8B",
        scope="full_776",
        label="Agentic Llama 8B",
        results_glob="data/results/legalbench_mini/reruns/agentic_llama_8b_tracefix/legalbench_mini_agentic_llama_8b_tracefix_off*.jsonl",
        csv_name="legalbench_mini_agentic_llama_8b_full_results.csv",
        sheet_name="legal_a8b",
    ),
    RunSpec(
        key="legal_trad_qwen_3b",
        benchmark="LegalBench-mini",
        mode="traditional",
        model="Qwen 2.5 3B",
        scope="full_776",
        label="Traditional Qwen 3B",
        results_glob="data/results/legalbench_mini/final/legalbench_mini_traditional_qwen25_3b_off*.jsonl",
        csv_name="legalbench_mini_traditional_qwen_3b_full_results.csv",
        sheet_name="legal_tq3b",
        notes="Extra model-choice run.",
    ),
    RunSpec(
        key="legal_trad_qwen_7b",
        benchmark="LegalBench-mini",
        mode="traditional",
        model="Qwen 2.5 7B",
        scope="full_776",
        label="Traditional Qwen 7B",
        results_glob="data/results/legalbench_mini/final/legalbench_mini_traditional_qwen25_7b_off*.jsonl",
        csv_name="legalbench_mini_traditional_qwen_7b_full_results.csv",
        sheet_name="legal_tq7b",
        notes="Extra model-choice run.",
    ),
    RunSpec(
        key="legal_agentic_qwen_3b",
        benchmark="LegalBench-mini",
        mode="agentic",
        model="Qwen 2.5 3B",
        scope="full_776",
        label="Agentic Qwen 3B",
        results_glob="data/results/legalbench_mini/final/legalbench_mini_agentic_qwen25_3b_off*.jsonl",
        csv_name="legalbench_mini_agentic_qwen_3b_full_results.csv",
        sheet_name="legal_aq3b",
        notes="Extra model-choice run.",
    ),
    RunSpec(
        key="legal_agentic_qwen_7b",
        benchmark="LegalBench-mini",
        mode="agentic",
        model="Qwen 2.5 7B",
        scope="full_776",
        label="Agentic Qwen 7B",
        results_glob="data/results/legalbench_mini/final/legalbench_mini_agentic_qwen25_7b_off*.jsonl",
        csv_name="legalbench_mini_agentic_qwen_7b_full_results.csv",
        sheet_name="legal_aq7b",
        notes="Extra model-choice run.",
    ),
    RunSpec(
        key="hotpot_trad_llama_3b",
        benchmark="HotpotQA",
        mode="traditional",
        model="Llama 3B",
        scope="full_1000",
        label="Traditional Llama 3B",
        results_glob="data/results/hotpotqa/reruns/traditional_llama_3b_shortanswerfix/hotpotqa_first1000_traditional_llama_3b_shortanswerfix_off*.jsonl",
        csv_name="hotpotqa_traditional_llama_3b_full_results.csv",
        sheet_name="hotpot_t3b",
    ),
    RunSpec(
        key="hotpot_trad_llama_8b",
        benchmark="HotpotQA",
        mode="traditional",
        model="Llama 8B",
        scope="full_1000",
        label="Traditional Llama 8B",
        results_glob="data/results/hotpotqa/reruns/traditional_llama_8b_shortanswerfix/hotpotqa_first1000_traditional_llama_8b_shortanswerfix_off*.jsonl",
        csv_name="hotpotqa_traditional_llama_8b_full_results.csv",
        sheet_name="hotpot_t8b",
    ),
    RunSpec(
        key="hotpot_agentic_llama_3b",
        benchmark="HotpotQA",
        mode="agentic",
        model="Llama 3B",
        scope="full_1000",
        label="Agentic Llama 3B",
        results_glob="data/results/hotpotqa/reruns/agentic_llama_3b_selectorfix/hotpotqa_first1000_agentic_llama_3b_selectorfix_off*.jsonl",
        csv_name="hotpotqa_agentic_llama_3b_full_results.csv",
        sheet_name="hotpot_a3b",
        notes="Extra model-choice run under corrected selector and retrieval-count setup.",
    ),
    RunSpec(
        key="hotpot_trad_qwen_3b",
        benchmark="HotpotQA",
        mode="traditional",
        model="Qwen 2.5 3B",
        scope="full_1000",
        label="Traditional Qwen 3B",
        results_glob="data/results/hotpotqa/final/hotpotqa_first1000_traditional_qwen25_3b_off*.jsonl",
        csv_name="hotpotqa_traditional_qwen_3b_full_results.csv",
        sheet_name="hotpot_tq3b",
        notes="Extra model-choice run.",
    ),
    RunSpec(
        key="hotpot_trad_qwen_7b",
        benchmark="HotpotQA",
        mode="traditional",
        model="Qwen 2.5 7B",
        scope="full_1000",
        label="Traditional Qwen 7B",
        results_glob="data/results/hotpotqa/final/hotpotqa_first1000_traditional_qwen25_7b_off*.jsonl",
        csv_name="hotpotqa_traditional_qwen_7b_full_results.csv",
        sheet_name="hotpot_tq7b",
        notes="Extra model-choice run.",
    ),
    RunSpec(
        key="hotpot_agentic_qwen_3b",
        benchmark="HotpotQA",
        mode="agentic",
        model="Qwen 2.5 3B",
        scope="full_1000",
        label="Agentic Qwen 3B",
        results_glob="data/results/hotpotqa/reruns/agentic_qwen_3b_selectorfix/hotpotqa_first1000_agentic_qwen25_3b_selectorfix_off*.jsonl",
        csv_name="hotpotqa_agentic_qwen_3b_full_results.csv",
        sheet_name="hotpot_aq3b",
        notes="Extra model-choice rerun with corrected selector and retrieval-count setup.",
    ),
    RunSpec(
        key="hotpot_agentic_llama_8b",
        benchmark="HotpotQA",
        mode="agentic",
        model="Llama 8B",
        scope="full_1000",
        label="Agentic Llama 8B",
        results_glob="data/results/hotpotqa/final/hotpotqa_first1000_agentic_8b_shortprompt_off*.jsonl",
        csv_name="hotpotqa_agentic_llama_8b_full_results.csv",
        sheet_name="hotpot_a8b",
    ),
    RunSpec(
        key="hotpot_agentic_qwen_7b",
        benchmark="HotpotQA",
        mode="agentic",
        model="Qwen 2.5 7B",
        scope="full_1000",
        label="Agentic Qwen 7B",
        results_glob="data/results/hotpotqa/reruns/agentic_qwen_7b_selectorfix/hotpotqa_first1000_agentic_qwen25_7b_selectorfix_off*.jsonl",
        csv_name="hotpotqa_agentic_qwen_7b_full_results.csv",
        sheet_name="hotpot_aq7b",
        notes="Extra model-choice rerun with corrected selector and retrieval-count setup.",
    ),
]

COMPACT_COLUMNS = [
    "row_index",
    "benchmark",
    "system_key",
    "system_label",
    "mode",
    "model",
    "scope",
    "query",
    "answer",
    "correct",
    "answer_f1",
    "evidence_recall",
    "topk_hit",
    "supporting_doc_recall",
    "latency_s",
    "retrieval_calls",
    "routing_type",
    "iterations",
    "retrieved_context_tokens",
    "prompt_tokens",
    "answer_tokens",
    "retrieval_latency_s",
    "generation_latency_s",
    "peak_rss_bytes",
    "error_flag",
    "error_message",
    "answer_selection_source",
    "answer_selection_reason",
    "first_pass_answer",
    "corrected_pass_answer",
    "run_id",
    "batch_id",
    "query_index_in_batch",
]

SUMMARY_COLUMNS = [
    "benchmark",
    "system_key",
    "system_label",
    "mode",
    "model",
    "scope",
    "rows",
    "source_files",
    "results_glob",
    "csv_file",
    "notes",
]


def load_jsonl_rows(pattern: str) -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    files = sorted(PROJECT_ROOT.glob(pattern))
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(normalize_trace_costs(json.loads(line)))
    return rows, files


def normalize_trace_costs(row: dict) -> dict:
    normalized = dict(row)
    trace = normalized.get("iteration_trace") or []
    if not trace:
        return normalized
    backup_used = any("||" in (step.get("retrieval_query") or "") for step in trace)
    retrieval_calls = normalized.get("retrieval_calls")
    iterations = normalized.get("iterations")
    if (
        backup_used
        and isinstance(retrieval_calls, (int, float))
        and not isinstance(retrieval_calls, bool)
        and isinstance(iterations, (int, float))
        and not isinstance(iterations, bool)
        and int(retrieval_calls) == int(iterations)
    ):
        normalized["retrieval_calls"] = int(normalized["retrieval_calls"]) + 1
    return normalized


def normalize_row(spec: RunSpec, row: dict, row_index: int) -> dict:
    return {
        "row_index": row_index,
        "benchmark": spec.benchmark,
        "system_key": spec.key,
        "system_label": spec.label,
        "mode": spec.mode,
        "model": spec.model,
        "scope": spec.scope,
        "query": row.get("query", ""),
        "answer": row.get("answer", ""),
        "correct": row.get("correct", ""),
        "answer_f1": row.get("answer_f1", ""),
        "evidence_recall": row.get("evidence_recall", ""),
        "topk_hit": row.get("topk_hit", ""),
        "supporting_doc_recall": row.get("supporting_doc_recall", ""),
        "latency_s": row.get("latency_s", ""),
        "retrieval_calls": row.get("retrieval_calls", ""),
        "routing_type": row.get("routing_type", ""),
        "iterations": row.get("iterations", ""),
        "retrieved_context_tokens": row.get("retrieved_context_tokens", ""),
        "prompt_tokens": row.get("prompt_tokens", ""),
        "answer_tokens": row.get("answer_tokens", ""),
        "retrieval_latency_s": row.get("retrieval_latency_s", ""),
        "generation_latency_s": row.get("generation_latency_s", ""),
        "peak_rss_bytes": row.get("peak_rss_bytes", ""),
        "error_flag": row.get("error_flag", ""),
        "error_message": row.get("error_message", ""),
        "answer_selection_source": row.get("answer_selection_source", ""),
        "answer_selection_reason": row.get("answer_selection_reason", ""),
        "first_pass_answer": row.get("first_pass_answer", ""),
        "corrected_pass_answer": row.get("corrected_pass_answer", ""),
        "run_id": row.get("run_id", ""),
        "batch_id": row.get("batch_id", ""),
        "query_index_in_batch": row.get("query_index_in_batch", ""),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def xml_cell(value) -> str:
    if value is None or value == "":
        return "<Cell/>"
    if isinstance(value, bool):
        return f'<Cell><Data ss:Type="String">{"TRUE" if value else "FALSE"}</Data></Cell>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<Cell><Data ss:Type="Number">{value}</Data></Cell>'
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return f'<Cell><Data ss:Type="String">{escape(text)}</Data></Cell>'


def sheet_xml(name: str, rows: list[dict], fieldnames: list[str]) -> str:
    xml = [f'<Worksheet ss:Name="{escape(name)}"><Table>']
    xml.append("<Row>" + "".join(xml_cell(col) for col in fieldnames) + "</Row>")
    for row in rows:
        xml.append("<Row>" + "".join(xml_cell(row.get(col, "")) for col in fieldnames) + "</Row>")
    xml.append("</Table></Worksheet>")
    return "".join(xml)


def workbook_xml(readme_rows: list[dict], summary_rows: list[dict], legal_all_rows: list[dict], hotpot_all_rows: list[dict], per_run_rows: dict[str, list[dict]]) -> str:
    parts = [
        '<?xml version="1.0"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" ',
        'xmlns:o="urn:schemas-microsoft-com:office:office" ',
        'xmlns:x="urn:schemas-microsoft-com:office:excel" ',
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet" ',
        'xmlns:html="http://www.w3.org/TR/REC-html40">',
    ]
    parts.append(sheet_xml("README", readme_rows, ["note"]))
    parts.append(sheet_xml("summary", summary_rows, SUMMARY_COLUMNS))
    parts.append(sheet_xml("legal_all", legal_all_rows, COMPACT_COLUMNS))
    parts.append(sheet_xml("hotpot_all", hotpot_all_rows, COMPACT_COLUMNS))
    for spec in RUN_SPECS:
        parts.append(sheet_xml(spec.sheet_name, per_run_rows[spec.key], COMPACT_COLUMNS))
    parts.append("</Workbook>")
    return "".join(parts)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    per_run_rows: dict[str, list[dict]] = {}
    summary_rows: list[dict] = []
    legal_all_rows: list[dict] = []
    hotpot_all_rows: list[dict] = []

    for spec in RUN_SPECS:
        raw_rows, files = load_jsonl_rows(spec.results_glob)
        normalized = [normalize_row(spec, row, i + 1) for i, row in enumerate(raw_rows)]
        per_run_rows[spec.key] = normalized
        summary_rows.append(
            {
                "benchmark": spec.benchmark,
                "system_key": spec.key,
                "system_label": spec.label,
                "mode": spec.mode,
                "model": spec.model,
                "scope": spec.scope,
                "rows": len(normalized),
                "source_files": len(files),
                "results_glob": spec.results_glob,
                "csv_file": spec.csv_name,
                "notes": spec.notes,
            }
        )
        write_csv(CSV_DIR / spec.csv_name, normalized, COMPACT_COLUMNS)
        if spec.benchmark == "LegalBench-mini":
            legal_all_rows.extend(normalized)
        elif spec.benchmark == "HotpotQA":
            hotpot_all_rows.extend(normalized)

    write_csv(CSV_DIR / "legalbench_mini_all_full_results.csv", legal_all_rows, COMPACT_COLUMNS)
    write_csv(CSV_DIR / "hotpotqa_all_full_results.csv", hotpot_all_rows, COMPACT_COLUMNS)
    write_csv(CSV_DIR / "full_results_summary.csv", summary_rows, SUMMARY_COLUMNS)

    readme_rows = [
        {"note": "Full benchmark result tables for thesis submission and Excel import."},
        {"note": "Each CSV contains one row per evaluated query."},
        {"note": "The workbook has separate sheets for combined benchmark views and each full run."},
        {"note": "Included runs: LegalBench-mini full Llama 3B traditional/agentic, LegalBench-mini full Llama 8B traditional/agentic, LegalBench-mini full Qwen 3B traditional/agentic, LegalBench-mini full Qwen 7B traditional/agentic, HotpotQA full Llama 3B/8B/agentic 8B, HotpotQA full Qwen 3B traditional/agentic, HotpotQA full Qwen 7B traditional/agentic."},
        {"note": "Excluded from this pack: probe runs, smoke runs, and preliminary Hotpot Qwen smoke20 because it is not a full benchmark run."},
        {"note": "Raw JSONL files with trace details remain under /Users/fadimardelli/local-rag-agents-benchmark/data/results/."},
    ]
    workbook = workbook_xml(readme_rows, summary_rows, legal_all_rows, hotpot_all_rows, per_run_rows)
    WORKBOOK_PATH.write_text(workbook, encoding="utf-8")

    readme_md = [
        "# Full Results Submission Pack\n\n",
        "This folder contains Excel-friendly exports of the full benchmark result tables for thesis submission.\n\n",
        "## Files\n",
        "- `csv/full_results_summary.csv`: run manifest and source mapping.\n",
        "- `csv/legalbench_mini_all_full_results.csv`: combined LegalBench-mini full-run rows.\n",
        "- `csv/hotpotqa_all_full_results.csv`: combined HotpotQA full-run rows.\n",
        "- `csv/*.csv`: one CSV per full run.\n",
        f"- `{WORKBOOK_PATH.name}`: multi-sheet Excel-compatible workbook (SpreadsheetML XML).\n",
        "\n## Notes\n",
        "- One row corresponds to one evaluated query.\n",
        "- The workbook excludes probe and smoke runs; only completed full benchmark runs are included.\n",
        "- Raw traced JSONL outputs remain in `data/results/` for deeper inspection.\n",
    ]
    (OUT_ROOT / "README.md").write_text("".join(readme_md), encoding="utf-8")

    print(f"Wrote CSV exports to {CSV_DIR}")
    print(f"Wrote workbook to {WORKBOOK_PATH}")


if __name__ == "__main__":
    main()
