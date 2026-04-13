import pandas as pd
from pathlib import Path
from src.indexing.embeddings import EmbeddingModel
from src.indexing.vector_store import build_faiss_index, save_faiss_index, save_metadata

# 1. Setup paths
CSV_PATH = "squad_t1_test.csv"
OUTPUT_DIR = Path("squad_index")
MODEL_NAME = "all-MiniLM-L6-v2" 

def run_indexing():
    print(f"--- Loading Dataset: {CSV_PATH} ---")
    if not Path(CSV_PATH).exists():
        print(f"ERROR: Could not find {CSV_PATH}. Make sure it is in your root folder.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # We embed the unique Wikipedia paragraphs
    contexts = df['context'].unique().tolist()
    print(f"Found {len(contexts)} unique paragraphs to index.")

    # 2. Initialize the Embedding Model (The 'Translator')
    print(f"--- Initializing Embedding Model: {MODEL_NAME} ---")
    encoder = EmbeddingModel(model_name=MODEL_NAME)

    # 3. Create the Vectors (Math-heavy part for the CPU)
    print("--- Encoding paragraphs into vectors... ---")
    vectors = encoder.encode(contexts)

    # 4. Build the FAISS Index (The 'Filing Cabinet')
    print("--- Building FAISS Index ---")
    index = build_faiss_index(vectors)

    # 5. Save everything with the "Big Four" metadata keys for Fadi's Retriever
    print(f"--- Saving index to folder: {OUTPUT_DIR} ---")
    
    # CRITICAL FIX: Aligning metadata with Fadi's Retriever requirements
    metadata = [
        {
            "text": str(c),                # Retriever looks for 'text'
            "start": 0,                    # Retriever looks for 'start'
            "end": len(str(c)),            # Retriever looks for 'end'
            "doc_path": "squad_v2_source"  # Retriever looks for 'doc_path'
        } for c in contexts
    ]
    
    save_faiss_index(index, OUTPUT_DIR / "index.faiss")
    save_metadata(metadata, OUTPUT_DIR / "metadata.jsonl")
    
    print("\nSUCCESS: SQuAD Index created without schema errors.")

if __name__ == "__main__":
    run_indexing()