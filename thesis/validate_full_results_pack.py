from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import export_full_results_pack as pack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / 'thesis' / 'submission_pack' / 'full_results' / 'csv'
WORKBOOK_PATH = PROJECT_ROOT / 'thesis' / 'submission_pack' / 'full_results' / 'benchmark_full_results_workbook.xml'

CHECK_FIELDS = [
    'query',
    'answer',
    'correct',
    'answer_f1',
    'evidence_recall',
    'topk_hit',
    'supporting_doc_recall',
    'latency_s',
    'retrieval_calls',
    'routing_type',
    'iterations',
    'retrieved_context_tokens',
    'prompt_tokens',
    'answer_tokens',
    'retrieval_latency_s',
    'generation_latency_s',
    'peak_rss_bytes',
    'error_flag',
    'error_message',
    'answer_selection_source',
    'answer_selection_reason',
    'first_pass_answer',
    'corrected_pass_answer',
    'run_id',
    'batch_id',
    'query_index_in_batch',
]


def normalize_for_compare(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'True' if value else 'False'
    return str(value)


def load_raw_rows(pattern: str):
    rows = []
    for path in sorted(PROJECT_ROOT.glob(pattern)):
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(pack.normalize_trace_costs(json.loads(line)))
    return rows


def load_csv_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    ET.parse(WORKBOOK_PATH)
    print(f'OK workbook XML parsed: {WORKBOOK_PATH}')

    legal_total = 0
    hotpot_total = 0

    for spec in pack.RUN_SPECS:
        raw_rows = load_raw_rows(spec.results_glob)
        csv_path = CSV_DIR / spec.csv_name
        csv_rows = load_csv_rows(csv_path)
        assert len(raw_rows) == len(csv_rows), f'Row count mismatch for {spec.key}: raw={len(raw_rows)} csv={len(csv_rows)}'

        for idx, (raw, csv_row) in enumerate(zip(raw_rows, csv_rows), start=1):
            assert csv_row['row_index'] == str(idx), f'row_index mismatch for {spec.key} row {idx}'
            assert csv_row['benchmark'] == spec.benchmark, f'benchmark mismatch for {spec.key} row {idx}'
            assert csv_row['system_key'] == spec.key, f'system_key mismatch for {spec.key} row {idx}'
            assert csv_row['system_label'] == spec.label, f'system_label mismatch for {spec.key} row {idx}'
            assert csv_row['mode'] == spec.mode, f'mode mismatch for {spec.key} row {idx}'
            assert csv_row['model'] == spec.model, f'model mismatch for {spec.key} row {idx}'
            assert csv_row['scope'] == spec.scope, f'scope mismatch for {spec.key} row {idx}'
            for field in CHECK_FIELDS:
                lhs = normalize_for_compare(raw.get(field))
                rhs = normalize_for_compare(csv_row.get(field, ''))
                assert lhs == rhs, f'{spec.key} row {idx} field {field!r} mismatch: raw={lhs!r} csv={rhs!r}'

        print(f'OK run {spec.key}: {len(raw_rows)} rows verified against {csv_path.name}')

        if spec.benchmark == 'LegalBench-mini':
            legal_total += len(raw_rows)
        elif spec.benchmark == 'HotpotQA':
            hotpot_total += len(raw_rows)

    legal_all = load_csv_rows(CSV_DIR / 'legalbench_mini_all_full_results.csv')
    hotpot_all = load_csv_rows(CSV_DIR / 'hotpotqa_all_full_results.csv')
    summary = load_csv_rows(CSV_DIR / 'full_results_summary.csv')

    assert len(legal_all) == legal_total, f'Legal combined count mismatch: expected {legal_total}, got {len(legal_all)}'
    assert len(hotpot_all) == hotpot_total, f'Hotpot combined count mismatch: expected {hotpot_total}, got {len(hotpot_all)}'
    assert len(summary) == len(pack.RUN_SPECS), f'Summary row mismatch: expected {len(pack.RUN_SPECS)}, got {len(summary)}'

    print(f'OK combined legal rows: {len(legal_all)}')
    print(f'OK combined hotpot rows: {len(hotpot_all)}')
    print(f'OK summary rows: {len(summary)}')
    print('All full-results exports verified successfully.')


if __name__ == '__main__':
    main()
