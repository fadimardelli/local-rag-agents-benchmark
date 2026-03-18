# Local RAG Benchmark for Legal QA (Traditional vs Agentic)

This repository implements a fully local benchmark framework for comparing:

- **Traditional RAG**: one retrieval pass, one answer generation.
- **Agentic RAG**: bounded iterative loop (retrieve -> assess -> refine -> retrieve), max 3 iterations.

The current implementation focuses on **T1/T2-style single-document legal QA** and includes an evaluation harness with latency, memory, retrieval, and evidence metrics.

## Scope

- Local-only inference (no cloud API calls in runtime pipeline)
- Dense retrieval with FAISS
- Llama.cpp-backed local generation (GGUF models)
- Evaluation on:
  - LegalBench-RAG ContractNLI subset (evidence recall focus)
  - Original ContractNLI (label accuracy + evidence recall)

## Tech Stack

- LLM runtime: `llama-cpp-python`
- Models:
  - `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
  - `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- Embeddings: `BAAI/bge-small-en-v1.5`
- Vector index: `FAISS IndexFlatIP` (cosine behavior via normalized embeddings)
- Python: `3.12`

## Repository Layout

- `/Users/fadimardelli/local-rag-agents-benchmark/src/config`
  - Global defaults (paths, model settings, chunking, retrieval)
- `/Users/fadimardelli/local-rag-agents-benchmark/src/data`
  - Dataset loaders and chunking utilities
- `/Users/fadimardelli/local-rag-agents-benchmark/src/indexing`
  - Embedding wrapper and FAISS helpers
- `/Users/fadimardelli/local-rag-agents-benchmark/src/rag`
  - Retriever, prompts, llama.cpp wrapper, traditional + agentic pipelines
- `/Users/fadimardelli/local-rag-agents-benchmark/src/eval`
  - Dataset iterators, metrics, evaluation harness
- `/Users/fadimardelli/local-rag-agents-benchmark/scripts`
  - Index building, run scripts, evaluation scripts, plotting
- `/Users/fadimardelli/local-rag-agents-benchmark/data`
  - Benchmarks, corpus, indexes, results

## Setup

1. Create and activate environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure model files exist (or use `--model-path` at runtime):

- `/Users/fadimardelli/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
- `/Users/fadimardelli/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`

4. Ensure datasets exist:

- LegalBench-RAG files in:
  - `/Users/fadimardelli/local-rag-agents-benchmark/data/benchmarks`
  - `/Users/fadimardelli/local-rag-agents-benchmark/data/corpus`
- Original ContractNLI files in:
  - `/Users/fadimardelli/local-rag-agents-benchmark/data/contract_nli_original`
  - expected split file for current eval: `test.json`

## Build Indexes

LegalBench-RAG ContractNLI index:

```bash
PYTHONPATH=. python scripts/build_index.py
```

Original ContractNLI index:

```bash
PYTHONPATH=. python scripts/build_index_contractnli_original.py
```

## Run Single Query

Traditional:

```bash
PYTHONPATH=. python scripts/run_traditional.py "Does the agreement allow verbally conveyed confidential information?" --model 8b --temperature 0.0 --max-tokens 256
```

Agentic:

```bash
PYTHONPATH=. python scripts/run_agentic.py "Does the agreement allow verbally conveyed confidential information?" --model 8b --temperature 0.0 --max-tokens 256
```

Both scripts print:

- Answer
- Latency (seconds)
- Peak RSS (MB)
- Average CPU usage (%)
- Retrieved chunk metadata

## Evaluate (Batch)

LegalBench-RAG ContractNLI:

```bash
PYTHONPATH=. python scripts/run_eval_contractnli.py --mode traditional --limit 50 --warmup --model 8b
PYTHONPATH=. python scripts/run_eval_contractnli.py --mode agentic --limit 50 --warmup --model 8b
```

Original ContractNLI (use label mode for accuracy):

```bash
PYTHONPATH=. python scripts/run_eval_contractnli_original.py --mode traditional --limit 50 --label-mode --warmup --model 8b
PYTHONPATH=. python scripts/run_eval_contractnli_original.py --mode agentic --limit 50 --label-mode --warmup --model 8b
```

### Export Results

Example with JSONL + CSV + summary CSV:

```bash
PYTHONPATH=. python scripts/run_eval_contractnli_original.py \
  --mode traditional \
  --limit 50 \
  --label-mode \
  --warmup \
  --model 8b \
  --output-jsonl data/results/orig_trad.jsonl \
  --output-csv data/results/orig_trad.csv \
  --summary-csv data/results/orig_trad_summary.csv
```

## Plotting

Generate comparison plots from two JSONL result files:

```bash
PYTHONPATH=. python scripts/plot_results.py \
  --a data/results/orig_trad.jsonl \
  --b data/results/orig_agentic.jsonl \
  --out-dir data/results/plots \
  --label-a Traditional \
  --label-b Agentic
```

## Metrics Currently Logged

- `answer_accuracy` (when labels exist)
- `evidence_recall_mean`
- `latency_median_s`
- `latency_p95_s`
- `peak_rss_bytes_max`
- `retrieval_calls_mean`
- `iterations_mean`
- `retrieved_context_chars_mean`
- `prompt_chars_mean`
- `answer_chars_mean`

## Reproducibility Notes

- Keep model, retriever settings, prompt format, and context budget fixed between methods.
- Prefer `--warmup` for stable timing.
- On macOS, check swap usage before long runs; heavy swap can invalidate latency comparisons.

```bash
sysctl vm.swapusage
```

## Status

- Implemented: T1/T2-style legal single-document benchmarking (traditional + agentic + evaluation).
- Planned next: T3/T4 multi-document integration (HotpotQA workflow).

## T3/T4 Workflow (HotpotQA)

1. Prepare HotpotQA validation artifacts:

```bash
PYTHONPATH=. python scripts/prepare_hotpotqa.py --split validation
```

2. Build HotpotQA index:

```bash
PYTHONPATH=. python scripts/build_index_hotpotqa.py
```

3. Evaluate Traditional vs Agentic on HotpotQA:

```bash
PYTHONPATH=. python scripts/run_eval_hotpotqa.py --mode traditional --limit 50 --warmup --model 8b
PYTHONPATH=. python scripts/run_eval_hotpotqa.py --mode agentic --limit 50 --warmup --model 8b
```

HotpotQA summary adds `supporting_doc_recall_mean` for multi-document retrieval quality.
