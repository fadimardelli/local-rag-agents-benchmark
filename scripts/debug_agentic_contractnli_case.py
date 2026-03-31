import argparse
from pathlib import Path

from src.config.defaults import LLAMA_GGUF_PATH_3B, LLAMA_GGUF_PATH_8B
from src.eval.datasets import iter_contractnli_cases_window
from src.eval.metrics import evidence_recall_multi_file
from src.rag import AgenticRAG


def fmt_bool(v: bool) -> str:
    return "yes" if v else "no"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--model", choices=["3b", "8b"], default="3b")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    case = next(iter_contractnli_cases_window(offset=args.offset, limit=1))
    question = args.query or case.query
    model_path: Path = LLAMA_GGUF_PATH_3B if args.model == "3b" else LLAMA_GGUF_PATH_8B

    rag = AgenticRAG(model_path=model_path, trace_iterations=True)
    out = rag.run(question, top_k=args.top_k)

    print("\nAgentic Debug\n")
    print(f"case_id: {case.case_id}")
    print(f"offset: {args.offset}")
    print(f"gold_spans: {case.gold_spans}")
    print("\noriginal_query:")
    print(question)

    for step in out.iteration_trace:
        print("\n---")
        print(f"iteration: {step.get('iteration_index')}")
        print(f"query_used: {step.get('query')}")
        print(f"retrieval_query: {step.get('retrieval_query')}")
        print(f"controller_decision: {step.get('controller_decision')}")
        print(f"controller_confidence: {step.get('controller_confidence')}")
        print(f"missing_aspect: {step.get('missing_aspect')}")
        print(f"refined_query: {step.get('refined_query')}")
        print(f"stop_reason: {step.get('stop_reason')}")
        print(f"controller_latency_s: {step.get('assess_latency_s')}")
        print("raw_controller_output:")
        print(step.get("raw_controller_output"))

        retrieved_refs = step.get("retrieved_refs", [])
        used_refs = step.get("used_refs", [])
        print(f"retrieved_count: {len(retrieved_refs)}")
        print(f"used_count: {len(used_refs)}")

        retrieved_chunks = [
            type("TmpChunk", (), ref | {"text": ""})() for ref in retrieved_refs
        ]
        used_chunks = [
            type("TmpChunk", (), ref | {"text": ""})() for ref in used_refs
        ]

        # evidence_recall_multi_file only reads doc_path/start/end
        topk_hit = evidence_recall_multi_file(retrieved_chunks, case.gold_spans)
        ev_hit = evidence_recall_multi_file(used_chunks, case.gold_spans)
        print(f"topk_hit_this_iteration: {fmt_bool(topk_hit)}")
        print(f"evidence_recall_this_iteration: {fmt_bool(ev_hit)}")

        if retrieved_refs:
            print("retrieved_refs:")
            for idx, ref in enumerate(retrieved_refs, start=1):
                print(
                    f"  {idx}. {ref['doc_path']} [{ref['start']}, {ref['end']}] score={ref.get('score')}"
                )

    print("\n=== Final ===")
    print(f"final_iterations: {out.iterations}")
    print(f"final_refined_queries: {out.refined_queries}")
    print(f"final_topk_hit: {fmt_bool(evidence_recall_multi_file(out.retrieved, case.gold_spans))}")
    print(f"final_evidence_recall: {fmt_bool(evidence_recall_multi_file(out.used, case.gold_spans))}")
    print(f"retrieval_latency_s: {out.retrieval_latency_s}")
    print(f"generation_latency_s: {out.generation_latency_s}")
    print("\nfinal_answer:")
    print(out.answer)


if __name__ == "__main__":
    main()
