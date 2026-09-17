from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from final_result_specs import RUN_SPECS, RunSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "thesis" / "submission_pack" / "full_results"
CSV_DIR = OUT_ROOT / "csv"
WORKBOOK_PATH = OUT_ROOT / "benchmark_full_results_workbook.xml"

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


def csv_name_for(spec: RunSpec) -> str:
    bench = "legalbench_mini" if spec.benchmark == "LegalBench-mini" else "hotpotqa"
    family = "llama" if spec.family == "Llama" else "qwen"
    size = f"{spec.size_b}b"
    return f"{bench}_{spec.mode}_{family}_{size}_full_results.csv"


def sheet_name_for(spec: RunSpec) -> str:
    bench = "legal" if spec.benchmark == "LegalBench-mini" else "hotpot"
    mode = "a" if spec.mode == "agentic" else "t"
    family = "l" if spec.family == "Llama" else "q"
    return f"{bench}_{mode}{family}{spec.size_b}"


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
        parts.append(sheet_xml(sheet_name_for(spec), per_run_rows[spec.key], COMPACT_COLUMNS))
    parts.append("</Workbook>")
    return "".join(parts)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    per_run_rows: dict[str, list[dict]] = {}
    summary_rows: list[dict] = []
    legal_all_rows: list[dict] = []
    hotpot_all_rows: list[dict] = []

    for spec in RUN_SPECS:
        raw_rows, files = load_jsonl_rows(spec.local_glob)
        normalized = [normalize_row(spec, row, i + 1) for i, row in enumerate(raw_rows)]
        per_run_rows[spec.key] = normalized
        csv_name = csv_name_for(spec)
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
                "results_glob": spec.local_glob,
                "csv_file": csv_name,
                "notes": spec.notes,
            }
        )
        write_csv(CSV_DIR / csv_name, normalized, COMPACT_COLUMNS)
        if spec.benchmark == "LegalBench-mini":
            legal_all_rows.extend(normalized)
        else:
            hotpot_all_rows.extend(normalized)

    write_csv(CSV_DIR / "legalbench_mini_all_full_results.csv", legal_all_rows, COMPACT_COLUMNS)
    write_csv(CSV_DIR / "hotpotqa_all_full_results.csv", hotpot_all_rows, COMPACT_COLUMNS)
    write_csv(CSV_DIR / "full_results_summary.csv", summary_rows, SUMMARY_COLUMNS)

    readme_rows = [
        {"note": "Full benchmark result tables for thesis submission and Excel import."},
        {"note": "Each CSV contains one row per evaluated query."},
        {"note": "The workbook has separate sheets for combined benchmark views and each final local run."},
        {"note": "Included runs: all 16 corrected local benchmark conditions (LegalBench-mini 8 conditions, HotpotQA 8 conditions)."},
        {"note": "Excluded from this pack: probe runs, smoke runs, subset-only checks, and pod-based robustness runs."},
        {"note": f"Raw JSONL files with trace details remain under {PROJECT_ROOT / 'data' / 'results'}."},
    ]
    workbook = workbook_xml(readme_rows, summary_rows, legal_all_rows, hotpot_all_rows, per_run_rows)
    WORKBOOK_PATH.write_text(workbook, encoding="utf-8")

    readme_md = [
        "# Full Results Submission Pack\n\n",
        "This folder contains Excel-friendly exports of the final local benchmark result tables for thesis submission.\n\n",
        "## Files\n",
        "- `csv/full_results_summary.csv`: run manifest and source mapping.\n",
        "- `csv/legalbench_mini_all_full_results.csv`: combined LegalBench-mini full-run rows.\n",
        "- `csv/hotpotqa_all_full_results.csv`: combined HotpotQA full-run rows.\n",
        "- `csv/*.csv`: one CSV per full run.\n",
        f"- `{WORKBOOK_PATH.name}`: multi-sheet Excel-compatible workbook (SpreadsheetML XML).\n",
        "\n## Notes\n",
        "- One row corresponds to one evaluated query.\n",
        "- The workbook excludes probe, smoke, subset-only, and pod-only robustness runs.\n",
        "- Raw traced JSONL outputs remain in `data/results/` for deeper inspection.\n",
    ]
    (OUT_ROOT / "README.md").write_text("".join(readme_md), encoding="utf-8")

    print(f"Wrote CSV exports to {CSV_DIR}")
    print(f"Wrote workbook to {WORKBOOK_PATH}")


if __name__ == "__main__":
    main()
