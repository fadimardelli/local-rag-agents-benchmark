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
