from pathlib import Path

PROJECT_ROOT = Path(".")
DATA_ROOT = PROJECT_ROOT / "data"
BENCHMARKS_DIR = DATA_ROOT / "benchmarks"
CORPUS_DIR = DATA_ROOT / "corpus"

LEGALBENCH_CONTRACTNLI_PATH = BENCHMARKS_DIR / "contractnli.json"

# Chunking defaults (character-based for simplicity and determinism)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Embedding / indexing defaults
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_NORMALIZE = True
INDEX_DIR = DATA_ROOT / "indexes"
CONTRACTNLI_INDEX_PATH = INDEX_DIR / "contractnli.faiss"
CONTRACTNLI_META_PATH = INDEX_DIR / "contractnli_meta.jsonl"

# Original ContractNLI (labeled) dataset
CONTRACTNLI_ORIG_DIR = DATA_ROOT / "contract_nli_original"
CONTRACTNLI_ORIG_TEST_PATH = CONTRACTNLI_ORIG_DIR / "test.json"
CONTRACTNLI_ORIG_INDEX_PATH = INDEX_DIR / "contractnli_original.faiss"
CONTRACTNLI_ORIG_META_PATH = INDEX_DIR / "contractnli_original_meta.jsonl"

# Retrieval defaults
RETRIEVAL_TOP_K = 5

# Agentic loop defaults
AGENTIC_MAX_ITERS = 3

# LLM defaults (llama.cpp)
LLAMA_GGUF_PATH_8B = Path("/Users/fadimardelli/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
LLAMA_GGUF_PATH_3B = Path("/Users/fadimardelli/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
LLAMA_GGUF_PATH = LLAMA_GGUF_PATH_8B
LLAMA_N_CTX = 4096
LLAMA_TEMPERATURE = 0.2
LLAMA_MAX_TOKENS = 512

# Context assembly (character budget, rough proxy for tokens)
CONTEXT_CHAR_BUDGET = 6000
