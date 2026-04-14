from pathlib import Path

PROJECT_ROOT = Path(".")
DATA_ROOT = PROJECT_ROOT / "data"
BENCHMARKS_DIR = DATA_ROOT / "benchmarks"
CORPUS_DIR = DATA_ROOT / "corpus"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"

LEGALBENCH_CONTRACTNLI_PATH = BENCHMARKS_DIR / "contractnli.json"
LEGALBENCH_MINI_PATH = BENCHMARKS_DIR / "legalbenchrag_mini.json"
LEGALBENCH_MINI_BALANCED_PATH = BENCHMARKS_DIR / "legalbenchrag_mini_balanced.json"
CONTRACTNLI_TASK_SPLIT_PATH = ANNOTATIONS_DIR / "contractnli_task_split.csv"

# Chunking defaults. Chunks are now assembled on text boundaries
# (line-aware for legal contracts, sentence-aware for prose) before
# falling back to smaller character windows inside oversized segments.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Embedding / indexing defaults
# Stronger drop-in sentence embedding model for retrieval ablations.
# Chosen over bge-small as the best quality/speed balance for local runs.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_NORMALIZE = True
INDEX_DIR = DATA_ROOT / "indexes"
CONTRACTNLI_INDEX_PATH = INDEX_DIR / "contractnli.faiss"
CONTRACTNLI_META_PATH = INDEX_DIR / "contractnli_meta.jsonl"

# Original ContractNLI (labeled) dataset
CONTRACTNLI_ORIG_DIR = DATA_ROOT / "contract_nli_original"
CONTRACTNLI_ORIG_TEST_PATH = CONTRACTNLI_ORIG_DIR / "test.json"
CONTRACTNLI_ORIG_INDEX_PATH = INDEX_DIR / "contractnli_original.faiss"
CONTRACTNLI_ORIG_META_PATH = INDEX_DIR / "contractnli_original_meta.jsonl"

# HotpotQA (multi-document) dataset artifacts
HOTPOTQA_DIR = DATA_ROOT / "hotpotqa"
HOTPOTQA_CASES_PATH = HOTPOTQA_DIR / "hotpotqa_validation_cases.jsonl"
HOTPOTQA_CORPUS_PATH = HOTPOTQA_DIR / "hotpotqa_validation_corpus.jsonl"
HOTPOTQA_INDEX_PATH = INDEX_DIR / "hotpotqa.faiss"
HOTPOTQA_META_PATH = INDEX_DIR / "hotpotqa_meta.jsonl"

# Retrieval defaults
RETRIEVAL_MODE = "dense"
RETRIEVAL_TOP_K = 5
HYBRID_DENSE_TOP_K = 20
HYBRID_BM25_TOP_K = 20
HYBRID_RRF_K = 60
RERANK_ENABLED = False
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATE_K = 20

# Retrieval-oriented query transformation defaults.
# "hyde" generates a short hypothetical clause/answer-like passage and embeds that
# instead of the raw user query. This is useful when the user's wording is much
# less document-like than the target evidence.
QUERY_TRANSFORM_MODE = "none"
QUERY_TRANSFORM_MAX_TOKENS = 96

# Agentic loop defaults
AGENTIC_MAX_ITERS = 2
AGENTIC_CORRECTIVE_TOP_K = 20
AGENTIC_CORRECTIVE_RRF_K = 60

# LLM defaults (llama.cpp)
LLAMA_GGUF_PATH_8B = Path("/Users/fadimardelli/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
LLAMA_GGUF_PATH_3B = Path("/Users/fadimardelli/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
LLAMA_GGUF_PATH = LLAMA_GGUF_PATH_8B
LLAMA_N_CTX = 4096
LLAMA_N_GPU_LAYERS = -1
LLAMA_SEED = 42
LLAMA_TEMPERATURE = 0.0
LLAMA_MAX_TOKENS = 512

# Context assembly (character budget, rough proxy for tokens)
CONTEXT_CHAR_BUDGET = 6000
