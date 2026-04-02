#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-traditional}"
MODEL="${2:-3b}"
BATCH_SIZE="${3:-25}"
PAUSE_SECONDS="${4:-600}"
RUN_ID="${5:-overnight_legalbench_mini_balanced_${MODE}_${MODEL}}"
TOP_K="${6:-10}"

if [[ "$MODE" != "traditional" && "$MODE" != "agentic" ]]; then
  echo "MODE must be 'traditional' or 'agentic'" >&2
  exit 1
fi

if [[ "$MODEL" != "3b" && "$MODEL" != "8b" ]]; then
  echo "MODEL must be '3b' or '8b'" >&2
  exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$BATCH_SIZE" -le 0 ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 1
fi

if ! [[ "$PAUSE_SECONDS" =~ ^[0-9]+$ ]] || [[ "$PAUSE_SECONDS" -lt 0 ]]; then
  echo "PAUSE_SECONDS must be a non-negative integer" >&2
  exit 1
fi

if ! [[ "$TOP_K" =~ ^[0-9]+$ ]] || [[ "$TOP_K" -le 0 ]]; then
  echo "TOP_K must be a positive integer" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="$ROOT_DIR/data/results"
mkdir -p "$RESULTS_DIR"

TOTAL_CASES="$(PYTHONPATH=. .venv/bin/python - <<'PY'
from src.eval.datasets import iter_legalbench_mini_balanced_cases
print(sum(1 for _ in iter_legalbench_mini_balanced_cases()))
PY
)"

LOG_PATH="$RESULTS_DIR/${RUN_ID}_overnight.log"

if command -v caffeinate >/dev/null 2>&1 && [[ "${UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export UNDER_CAFFEINATE=1
  exec caffeinate -i "$0" "$MODE" "$MODEL" "$BATCH_SIZE" "$PAUSE_SECONDS" "$RUN_ID" "$TOP_K"
fi

run_batches() {
  echo "Run ID: $RUN_ID"
  echo "Mode: $MODE"
  echo "Model: $MODEL"
  echo "Batch size: $BATCH_SIZE"
  echo "Pause seconds: $PAUSE_SECONDS"
  echo "Top-k: $TOP_K"
  echo "Retrieval mode: hybrid"
  echo "Total cases: $TOTAL_CASES"
  echo "Started at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  batch_num=0
  summary_files=()

  for ((offset=0; offset<TOTAL_CASES; offset+=BATCH_SIZE)); do
    batch_num=$((batch_num + 1))
    batch_id="b${BATCH_SIZE}_off${offset}"
    jsonl_path="$RESULTS_DIR/${RUN_ID}_off${offset}.jsonl"
    csv_path="$RESULTS_DIR/${RUN_ID}_off${offset}.csv"
    summary_path="$RESULTS_DIR/${RUN_ID}_off${offset}_summary.csv"
    summary_files+=("$summary_path")

    echo
    echo "=== Batch $batch_num: offset=$offset limit=$BATCH_SIZE ==="
    PYTHONPATH=. .venv/bin/python scripts/run_eval_legalbench_mini_balanced.py \
      --mode "$MODE" \
      --offset "$offset" \
      --limit "$BATCH_SIZE" \
      --warmup \
      --model "$MODEL" \
      --top-k "$TOP_K" \
      --query-transform-mode none \
      --retrieval-mode hybrid \
      --hybrid-dense-top-k 20 \
      --hybrid-bm25-top-k 20 \
      --hybrid-rrf-k 60 \
      --run-id "$RUN_ID" \
      --batch-id "$batch_id" \
      --summary-view all \
      --output-jsonl "$jsonl_path" \
      --output-csv "$csv_path" \
      --summary-csv "$summary_path"

    next_offset=$((offset + BATCH_SIZE))
    if (( next_offset < TOTAL_CASES && PAUSE_SECONDS > 0 )); then
      echo "Sleeping for $PAUSE_SECONDS seconds..."
      sleep "$PAUSE_SECONDS"
    fi
  done

  echo
  echo "=== Calibration Aggregate ==="
  PYTHONPATH=. .venv/bin/python scripts/analyze_batch_calibration.py "${summary_files[@]}" || true
  echo
  echo "Finished at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

run_batches | tee "$LOG_PATH"
