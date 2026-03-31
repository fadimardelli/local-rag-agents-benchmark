import argparse
from pathlib import Path

from src.config.defaults import LLAMA_GGUF_PATH_3B, LLAMA_GGUF_PATH_8B
from src.eval.datasets import iter_contractnli_original_cases
from src.eval.metrics import answer_correctness, evidence_recall_multi
from src.rag import AgenticRAG, MultiActionAgenticRAG, Retriever
from src.config.defaults import CONTRACTNLI_ORIG_INDEX_PATH, CONTRACTNLI_ORIG_META_PATH


def fmt_bool(v: bool | None) -> str:
    if v is None:
        return "n/a"
    return "yes" if v else "no"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--model", choices=["3b", "8b"], default="3b")
    parser.add_argument("--workflow", choices=["legacy", "multi_action"], default="multi_action")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--free-form",
        action="store_true",
        help="Use free-form answering instead of ContractNLI label mode.",
    )
    args = parser.parse_args()

    case = next(iter_contractnli_original_cases(offset=args.offset, limit=1))
    question = args.query or case.query
    model_path: Path = LLAMA_GGUF_PATH_3B if args.model == "3b" else LLAMA_GGUF_PATH_8B
    retriever = Retriever(
        index_path=CONTRACTNLI_ORIG_INDEX_PATH,
        meta_path=CONTRACTNLI_ORIG_META_PATH,
    )
    doc_path = retriever.resolve_doc_path(case.file_name)

    rag = (
        MultiActionAgenticRAG(retriever=retriever, model_path=model_path, trace_iterations=True)
        if args.workflow == "multi_action"
        else AgenticRAG(retriever=retriever, model_path=model_path, trace_iterations=True)
    )
    out = (
        rag.run(question, top_k=args.top_k, doc_path=doc_path)
        if args.free_form
        else rag.run_label(question, top_k=args.top_k, doc_path=doc_path)
    )

    print("\nAgentic Debug (ContractNLI Original)\n")
    print(f"offset: {args.offset}")
    print(f"gold_file: {case.file_name}")
    print(f"resolved_doc_path: {doc_path}")
    print(f"gold_label: {case.gold_label}")
    print(f"gold_spans: {case.gold_spans}")
    print(f"label_mode: {fmt_bool(not args.free_form)}")
    print("\noriginal_query:")
    print(question)

    for step in out.iteration_trace:
        print("\n---")
        print(f"iteration: {step.get('iteration_index')}")
        print(f"action: {step.get('action')}")
        print(f"controller_query: {step.get('controller_query', step.get('query'))}")
        print(f"controller_retrieval_query: {step.get('controller_retrieval_query', step.get('retrieval_query'))}")
        print(f"focused_document: {step.get('focused_document')}")
        print(f"document_index: {step.get('document_index')}")
        print(f"next_query: {step.get('next_query')}")
        print(f"result_query: {step.get('result_query')}")
        print(f"result_retrieval_query: {step.get('result_retrieval_query')}")
        print(f"result_focused_document: {step.get('result_focused_document')}")
        print(f"changed_retrieval: {step.get('changed_retrieval')}")
        print(f"source_guard_applied: {step.get('source_guard_applied')}")
        print(f"stop_guard_applied: {step.get('stop_guard_applied')}")
        print(f"anchor_source_score: {step.get('anchor_source_score')}")
        print(f"best_source_score: {step.get('best_source_score')}")
        print(f"rationale: {step.get('rationale')}")
        print(f"stop_reason: {step.get('stop_reason')}")
        print(f"controller_latency_s: {step.get('controller_latency_s', step.get('assess_latency_s'))}")
        print("raw_controller_output:")
        print(step.get("raw_controller_output"))

        controller_refs = step.get("controller_retrieved_refs", step.get("retrieved_refs", []))
        controller_used_refs = step.get("controller_used_refs", step.get("used_refs", []))
        result_refs = step.get("result_retrieved_refs", controller_refs)
        result_used_refs = step.get("result_used_refs", controller_used_refs)
        print(f"controller_retrieved_count: {len(controller_refs)}")
        print(f"controller_used_count: {len(controller_used_refs)}")
        print(f"result_retrieved_count: {len(result_refs)}")
        print(f"result_used_count: {len(result_used_refs)}")

        controller_chunks = [type("TmpChunk", (), ref | {"text": ""})() for ref in controller_refs]
        controller_used_chunks = [type("TmpChunk", (), ref | {"text": ""})() for ref in controller_used_refs]
        result_chunks = [type("TmpChunk", (), ref | {"text": ""})() for ref in result_refs]
        result_used_chunks = [type("TmpChunk", (), ref | {"text": ""})() for ref in result_used_refs]

        controller_topk_hit = evidence_recall_multi(controller_chunks, case.file_name, case.gold_spans)
        controller_ev_hit = evidence_recall_multi(controller_used_chunks, case.file_name, case.gold_spans)
        result_topk_hit = evidence_recall_multi(result_chunks, case.file_name, case.gold_spans)
        result_ev_hit = evidence_recall_multi(result_used_chunks, case.file_name, case.gold_spans)
        print(f"controller_topk_hit: {fmt_bool(controller_topk_hit)}")
        print(f"controller_evidence_recall: {fmt_bool(controller_ev_hit)}")
        print(f"result_topk_hit: {fmt_bool(result_topk_hit)}")
        print(f"result_evidence_recall: {fmt_bool(result_ev_hit)}")

        if controller_refs:
            print("controller_retrieved_refs:")
            for idx, ref in enumerate(controller_refs, start=1):
                print(f"  {idx}. {ref['doc_path']} [{ref['start']}, {ref['end']}] score={ref.get('score')}")
        if result_refs:
            print("result_retrieved_refs:")
            for idx, ref in enumerate(result_refs, start=1):
                print(f"  {idx}. {ref['doc_path']} [{ref['start']}, {ref['end']}] score={ref.get('score')}")

    print("\n=== Final ===")
    print(f"final_controller_iterations: {out.iterations}")
    if hasattr(out, "retrieval_steps"):
        print(f"final_retrieval_steps: {getattr(out, 'retrieval_steps')}")
    print(f"final_refined_queries: {out.refined_queries}")
    if hasattr(out, "action_history"):
        print(f"final_retrieval_action_history: {getattr(out, 'action_history')}")
    print(f"final_topk_hit: {fmt_bool(evidence_recall_multi(out.retrieved, case.file_name, case.gold_spans))}")
    print(f"final_evidence_recall: {fmt_bool(evidence_recall_multi(out.used, case.file_name, case.gold_spans))}")
    print(f"final_label_correct: {fmt_bool(answer_correctness(out.answer, case.gold_label))}")
    print(f"retrieval_latency_s: {out.retrieval_latency_s}")
    print(f"generation_latency_s: {out.generation_latency_s}")
    print("\nfinal_answer:")
    print(out.answer)


if __name__ == "__main__":
    main()
