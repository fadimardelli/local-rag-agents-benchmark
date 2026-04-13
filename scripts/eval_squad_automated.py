import time
import psutil
import pandas as pd
from pathlib import Path
from src.rag import TraditionalRAG, AgenticRAG, Retriever
from src.eval.squad_loader import iter_squad_cases

# Paths to your new SQuAD index
SQUAD_INDEX = Path("squad_index/index.faiss")
SQUAD_META = Path("squad_index/metadata.jsonl")

def is_hallucination(answer, label):
    # If the question is unanswerable, but the AI gives a long, confident answer
    # that doesn't say "I don't know" or "Not mentioned", it's a hallucination.
    if label == "Unanswerable":
        negative_keywords = ["not mentioned", "don't know", "no information", "cannot answer", "not found"]
        if not any(word in answer.lower() for word in negative_keywords):
            return True
    return False

def run_automated_benchmark(mode="traditional"):
    print(f"--- Starting {mode.upper()} Benchmark ---")
    retriever = Retriever(index_path=SQUAD_INDEX, meta_path=SQUAD_META)
    rag = TraditionalRAG(retriever=retriever) if mode == "traditional" else AgenticRAG(retriever=retriever)
    
    results = []
    cases = list(iter_squad_cases("squad_t1_test.csv"))

    for i, case in enumerate(cases):
        print(f"[{i+1}/100] Mode: {mode} | Type: {case.label}")
        
        start_time = time.time()
        cpu_start = psutil.cpu_percent(interval=None)
        
        output = rag.run(case.query)
        
        latency = time.time() - start_time
        cpu_end = psutil.cpu_percent(interval=None)
        
        # New: Logic to check if AI lied
        hallucinated = is_hallucination(output.answer, case.label)
        
        results.append({
            "mode": mode,
            "type": case.label,
            "query": case.query,
            "answer": output.answer,
            "latency_s": round(latency, 2),
            "cpu_avg": (cpu_start + cpu_end) / 2,
            "iterations": getattr(output, 'iterations', 1),
            "hallucinated": hallucinated
        })

    return results

# Run and Save
results_data = run_automated_benchmark("traditional") + run_automated_benchmark("agentic")
pd.DataFrame(results_data).to_csv("squad_final_eval.csv", index=False)
print("Done! Check squad_final_eval.csv")