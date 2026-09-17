# Local RAG Benchmark

This repository contains the code and supporting artifacts for a fully local benchmark comparing a traditional one-pass retrieval-augmented generation (RAG) pipeline with a bounded corrective RAG pipeline.

The main benchmark settings are:

- `LegalBench-mini`: source-focused legal evidence retrieval
- `HotpotQA`: closed-corpus multi-hop question answering

The project studies when corrective retrieval improves quality, when a stronger one-pass baseline is sufficient, and what local efficiency costs accompany correction.

## Repository Structure

- `src/`: core implementation
- `scripts/`: command-line entry points for preparation, indexing, and evaluation
- `experiments/`: supplementary analyses
- `tests/`: lightweight inspection helpers
- `thesis/`: reporting exports, figures, and thesis-oriented artifacts

## Core Pipelines

### Traditional one-pass RAG

- retrieve once
- build context once
- generate one answer

### Bounded corrective RAG

- retrieve an initial context
- assess whether the evidence is sufficient
- optionally rewrite the query
- retrieve once more
- generate the final answer

The corrective pipeline is bounded to at most one additional retrieval step.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected local data locations:

- `data/benchmarks/`
- `data/corpus/`
- `data/hotpotqa/`

## Index Construction

```bash
PYTHONPATH=. python scripts/build_index.py
PYTHONPATH=. python scripts/prepare_hotpotqa.py --split validation
PYTHONPATH=. python scripts/build_index_hotpotqa.py
```

## Main Evaluation Entry Points

LegalBench-mini:

```bash
PYTHONPATH=. python scripts/run_eval_legalbench_mini.py --mode traditional --limit 50 --warmup --model 8b
PYTHONPATH=. python scripts/run_eval_legalbench_mini.py --mode agentic --limit 50 --warmup --model 8b --trace-agentic
```

HotpotQA:

```bash
PYTHONPATH=. python scripts/run_eval_hotpotqa.py --mode traditional --limit 50 --warmup --model 8b
PYTHONPATH=. python scripts/run_eval_hotpotqa.py --mode agentic --limit 50 --warmup --model 8b --trace-agentic
```

## Documentation

- [Master's thesis](./docs/Thesis_Paper.pdf) — "Investigating Locally Deployed LLM Agentic Workflows" (Group T, KU Leuven). Full write-up of the motivation, methodology, and results behind this benchmark.
- [Defense presentation](./docs/Defence_Presentation.pdf) — "Evaluating Retrieval-Augmented Generation Frameworks Under Local Deployment Constraints," the slides used for the thesis defense.

## Notes

- The repository is designed for fully local inference at runtime.
- Large benchmark artifacts, indexes, and result files are maintained as local research assets rather than clean GitHub source files.
- Folder-specific `README.md` files provide additional navigation where useful, but the main project description should live in this root `README.md`.
