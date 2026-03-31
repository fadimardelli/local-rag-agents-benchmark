from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from typing import List, Optional

from src.config.defaults import (
    AGENTIC_MAX_ITERS,
    CONTEXT_CHAR_BUDGET,
    LLAMA_GGUF_PATH,
    LLAMA_MAX_TOKENS,
    LLAMA_N_CTX,
    LLAMA_N_GPU_LAYERS,
    LLAMA_SEED,
    LLAMA_TEMPERATURE,
    QUERY_TRANSFORM_MODE,
    RETRIEVAL_TOP_K,
)
from src.rag.context_builder import build_context
from src.rag.llama_cpp import LlamaCppModel
from src.rag.prompts import (
    CONTROLLER_DECISION_SYSTEM_PROMPT,
    CONTROLLER_REWRITE_SYSTEM_PROMPT,
    HOTPOT_SYSTEM_PROMPT,
    LABEL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_controller_prompt,
    build_rewrite_prompt,
    build_user_prompt,
)
from src.rag.query_transform import build_retrieval_query
from src.rag.retriever import RetrievedChunk, Retriever


CONTROLLER_DECISION_MAX_TOKENS = 48
CONTROLLER_REWRITE_MAX_TOKENS = 48
CONTROLLER_TEMPERATURE = 0.0


@dataclass(frozen=True)
class AgenticResult:
    answer: str
    retrieved: List[RetrievedChunk]
    used: List[RetrievedChunk]
    iterations: int
    refined_queries: List[str]
    retrieval_latency_s: float
    generation_latency_s: float
    retrieved_context_tokens: int
    prompt_tokens: int
    answer_tokens: int
    iteration_trace: List[dict]


class AgenticRAG:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        model: Optional[LlamaCppModel] = None,
        model_path: Optional[Path] = None,
        trace_iterations: bool = False,
    ):
        self.retriever = retriever or Retriever()
        self.trace_iterations = trace_iterations
        self.model = model or LlamaCppModel(
            model_path=model_path or LLAMA_GGUF_PATH,
            n_ctx=LLAMA_N_CTX,
            n_gpu_layers=LLAMA_N_GPU_LAYERS,
            seed=LLAMA_SEED,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
        )
        self.query_transform_mode = QUERY_TRANSFORM_MODE

    def _prepare_retrieval_query(self, question: str) -> tuple[str, float]:
        transformed = build_retrieval_query(
            question=question,
            model=self.model,
            mode=self.query_transform_mode,
        )
        return transformed.retrieval_query, transformed.latency_s

    @staticmethod
    def _chunk_ref(chunk: RetrievedChunk) -> dict:
        return {
            "doc_path": chunk.doc_path,
            "start": chunk.start,
            "end": chunk.end,
            "score": chunk.score,
        }

    def _make_iteration_entry(
        self,
        *,
        iteration_index: int,
        query: str,
        retrieval_query: str,
        retrieved: List[RetrievedChunk],
        used: List[RetrievedChunk],
        context: str,
        is_sufficient: bool,
        assess_latency_s: float,
        refine_latency_s: float = 0.0,
        refined_query: Optional[str] = None,
        stop_reason: Optional[str] = None,
        controller_decision: Optional[str] = None,
        controller_confidence: Optional[int] = None,
        missing_aspect: Optional[str] = None,
        raw_controller_output: Optional[str] = None,
    ) -> dict:
        return {
            "iteration_index": iteration_index,
            "query": query,
            "retrieval_query": retrieval_query,
            "retrieved_refs": [self._chunk_ref(chunk) for chunk in retrieved],
            "used_refs": [self._chunk_ref(chunk) for chunk in used],
            "context_chars": len(context),
            "is_sufficient": is_sufficient,
            "assess_latency_s": assess_latency_s,
            "refine_latency_s": refine_latency_s,
            "refined_query": refined_query,
            "stop_reason": stop_reason,
            "controller_decision": controller_decision,
            "controller_confidence": controller_confidence,
            "missing_aspect": missing_aspect,
            "raw_controller_output": raw_controller_output,
        }

    @staticmethod
    def _chunk_signature(chunks: List[RetrievedChunk]) -> tuple[tuple[str, int, int], ...]:
        return tuple((chunk.doc_path, chunk.start, chunk.end) for chunk in chunks)

    @staticmethod
    def _normalize_optional_text(text: Optional[str]) -> str:
        if text is None:
            return "NONE"
        cleaned = text.strip()
        return cleaned if cleaned else "NONE"

    @staticmethod
    def _parse_decision_output(text: str) -> str:
        cleaned = text.strip().upper()
        if cleaned == "STOP":
            return "STOP"
        if cleaned == "RETRY":
            return "RETRY"
        first_word = cleaned.split()[0] if cleaned else ""
        return "STOP" if first_word == "STOP" else "RETRY"

    def _parse_rewrite_output(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "NONE"
        # Strip simple quoting if present.
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()
        return self._normalize_optional_text(cleaned)

    @staticmethod
    def _entity_hints(question: str) -> str:
        prefix = question.split(";", 1)[0].strip()
        if prefix.lower().startswith("consider "):
            prefix = prefix[len("consider ") :].strip()
        if prefix and prefix != question:
            return prefix

        entities = []
        for match in re.finditer(r"\b(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Za-z]*[A-Z][A-Za-z]*)\b", question):
            token = match.group(0)
            if token.lower() in {"does", "the", "agreement", "document", "question", "consider"}:
                continue
            entities.append(token)
        deduped = []
        for token in entities:
            if token not in deduped:
                deduped.append(token)
        return ", ".join(deduped[:8]) if deduped else "NONE"

    @staticmethod
    def _controller_context(chunks: List[RetrievedChunk], char_budget: int) -> str:
        parts: List[str] = []
        total = 0
        for idx, chunk in enumerate(chunks, start=1):
            source = Path(chunk.doc_path).name
            block = (
                f"[Chunk {idx}]\n"
                f"Source: {source}\n"
                f"Span: {chunk.start}-{chunk.end}\n"
                f"Text: {chunk.text.strip()}\n"
            )
            add_len = len(block) + (2 if parts else 0)
            if total + add_len > char_budget:
                break
            if parts:
                parts.append("\n\n")
            parts.append(block)
            total += add_len
        return "".join(parts)

    def _decision_step(self, question: str, context: str) -> tuple[str, float, str]:
        user_prompt = build_controller_prompt(context=context, question=question)
        started = perf_counter()
        result = self.model.generate(
            CONTROLLER_DECISION_SYSTEM_PROMPT,
            user_prompt,
            temperature=CONTROLLER_TEMPERATURE,
            max_tokens=CONTROLLER_DECISION_MAX_TOKENS,
        )
        latency_s = perf_counter() - started
        decision = self._parse_decision_output(result)
        return decision, latency_s, result

    def _rewrite_step(self, question: str, context: str) -> tuple[str, float, str]:
        entity_hints = self._entity_hints(question)
        user_prompt = build_rewrite_prompt(
            question=question,
            entity_hints=entity_hints,
            context=context,
        )
        started = perf_counter()
        result = self.model.generate(
            CONTROLLER_REWRITE_SYSTEM_PROMPT,
            user_prompt,
            temperature=CONTROLLER_TEMPERATURE,
            max_tokens=CONTROLLER_REWRITE_MAX_TOKENS,
        )
        latency_s = perf_counter() - started
        rewrite = self._parse_rewrite_output(result)

        original_norm = self._normalize_query(question)
        rewrite_norm = self._normalize_query(rewrite)
        if rewrite == "NONE" or not rewrite_norm or rewrite_norm == original_norm:
            return "NONE", latency_s, result

        entity_hints = self._entity_hints(question)
        if entity_hints != "NONE":
            hint_tokens = [self._normalize_query(tok) for tok in re.split(r"[,\s]+", entity_hints) if tok.strip()]
            # Require the rewrite to preserve at least one meaningful entity token when such hints exist.
            if hint_tokens and not any(tok and tok in rewrite_norm for tok in hint_tokens):
                return "NONE", latency_s, result

        if any(token in rewrite for token in ("AND", " OR ", "(", ")", '"', "'")):
            return "NONE", latency_s, result

        return rewrite, latency_s, result

    @staticmethod
    def _normalize_query(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _run_impl(self, question: str, answer_system_prompt: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        refined_queries: List[str] = []
        retrieved: List[RetrievedChunk] = []
        used: List[RetrievedChunk] = []
        current_query = question
        retrieval_latency_s = 0.0
        generation_latency_s = 0.0
        iteration_trace: List[dict] = []
        previous_signature: Optional[tuple[tuple[str, int, int], ...]] = None
        best_stop_retrieved: List[RetrievedChunk] = []
        best_stop_used: List[RetrievedChunk] = []
        best_stop_context = ""

        for i in range(AGENTIC_MAX_ITERS):
            retrieval_started = perf_counter()
            retrieval_query, _ = self._prepare_retrieval_query(current_query)
            retrieved = self.retriever.retrieve(retrieval_query, k=top_k)
            retrieval_latency_s += perf_counter() - retrieval_started
            context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
            current_signature = self._chunk_signature(retrieved)
            controller_context = self._controller_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)

            decision, controller_latency_s, raw_controller_output = self._decision_step(
                question, controller_context
            )
            generation_latency_s += controller_latency_s

            if decision == "STOP":
                if not best_stop_context:
                    best_stop_retrieved = list(retrieved)
                    best_stop_used = list(used)
                    best_stop_context = context
                if self.trace_iterations:
                    iteration_trace.append(
                        self._make_iteration_entry(
                            iteration_index=i + 1,
                            query=current_query,
                            retrieval_query=retrieval_query,
                            retrieved=retrieved,
                            used=used,
                            context=context,
                            is_sufficient=True,
                            assess_latency_s=controller_latency_s,
                            stop_reason="sufficient",
                            controller_decision=decision,
                            controller_confidence=None,
                            missing_aspect="NONE",
                            raw_controller_output=raw_controller_output,
                        )
                    )
                break

            if i < AGENTIC_MAX_ITERS - 1:
                refined, rewrite_latency_s, raw_rewrite_output = self._rewrite_step(question, controller_context)
                generation_latency_s += rewrite_latency_s
                if refined == "NONE":
                    if self.trace_iterations:
                        iteration_trace.append(
                            self._make_iteration_entry(
                                iteration_index=i + 1,
                                query=current_query,
                                retrieval_query=retrieval_query,
                                retrieved=retrieved,
                                used=used,
                                context=context,
                                is_sufficient=False,
                                assess_latency_s=controller_latency_s,
                                refine_latency_s=rewrite_latency_s,
                                refined_query=refined,
                                stop_reason="empty_refinement",
                                controller_decision=decision,
                                controller_confidence=None,
                                missing_aspect="NONE",
                                raw_controller_output=f"DECISION>> {raw_controller_output}\nREWRITE>> {raw_rewrite_output}",
                            )
                        )
                    break
                if self._normalize_query(refined) == self._normalize_query(current_query):
                    if self.trace_iterations:
                        iteration_trace.append(
                            self._make_iteration_entry(
                                iteration_index=i + 1,
                                query=current_query,
                                retrieval_query=retrieval_query,
                                retrieved=retrieved,
                                used=used,
                                context=context,
                                is_sufficient=False,
                                assess_latency_s=controller_latency_s,
                                refine_latency_s=rewrite_latency_s,
                                refined_query=refined,
                                stop_reason="unchanged_refinement",
                                controller_decision=decision,
                                controller_confidence=None,
                                missing_aspect="NONE",
                                raw_controller_output=f"DECISION>> {raw_controller_output}\nREWRITE>> {raw_rewrite_output}",
                            )
                        )
                    break
                if previous_signature is not None and current_signature == previous_signature:
                    if self.trace_iterations:
                        iteration_trace.append(
                            self._make_iteration_entry(
                                iteration_index=i + 1,
                                query=current_query,
                                retrieval_query=retrieval_query,
                                retrieved=retrieved,
                                used=used,
                                context=context,
                                is_sufficient=False,
                                assess_latency_s=controller_latency_s,
                                refine_latency_s=rewrite_latency_s,
                                refined_query=refined,
                                stop_reason="repeated_retrieval",
                                controller_decision=decision,
                                controller_confidence=None,
                                missing_aspect="NONE",
                                raw_controller_output=f"DECISION>> {raw_controller_output}\nREWRITE>> {raw_rewrite_output}",
                            )
                        )
                    break
                if self.trace_iterations:
                    iteration_trace.append(
                        self._make_iteration_entry(
                            iteration_index=i + 1,
                            query=current_query,
                            retrieval_query=retrieval_query,
                            retrieved=retrieved,
                            used=used,
                            context=context,
                            is_sufficient=False,
                            assess_latency_s=controller_latency_s,
                            refine_latency_s=rewrite_latency_s,
                            refined_query=refined,
                            stop_reason="continue",
                            controller_decision=decision,
                            controller_confidence=None,
                            missing_aspect="NONE",
                            raw_controller_output=f"DECISION>> {raw_controller_output}\nREWRITE>> {raw_rewrite_output}",
                        )
                    )
                refined_queries.append(refined)
                current_query = refined
                previous_signature = current_signature
            elif self.trace_iterations:
                iteration_trace.append(
                    self._make_iteration_entry(
                        iteration_index=i + 1,
                        query=current_query,
                        retrieval_query=retrieval_query,
                        retrieved=retrieved,
                        used=used,
                        context=context,
                        is_sufficient=False,
                        assess_latency_s=controller_latency_s,
                        stop_reason="max_iterations_reached",
                        controller_decision=decision,
                        controller_confidence=None,
                        missing_aspect="NONE",
                        raw_controller_output=raw_controller_output,
                    )
                )

        final_context = best_stop_context or context
        final_retrieved = best_stop_retrieved or retrieved
        final_used = best_stop_used or used

        user_prompt = build_user_prompt(context=final_context, question=question)
        generation_started = perf_counter()
        final_system_prompt = answer_system_prompt
        if not best_stop_context:
            final_system_prompt = (
                answer_system_prompt
                + " The retrieval controller did not confirm sufficient evidence. "
                + "If the context does not directly support the answer, say you don't know."
            )
        answer = self.model.generate(final_system_prompt, user_prompt)
        generation_latency_s += perf_counter() - generation_started

        return AgenticResult(
            answer=answer,
            retrieved=final_retrieved,
            used=final_used,
            iterations=len(refined_queries) + 1,
            refined_queries=refined_queries,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(final_context),
            prompt_tokens=self.model.count_tokens(final_system_prompt) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
            iteration_trace=iteration_trace,
        )

    def run(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        return self._run_impl(question=question, answer_system_prompt=SYSTEM_PROMPT, top_k=top_k)

    def run_label(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        return self._run_impl(question=question, answer_system_prompt=LABEL_SYSTEM_PROMPT, top_k=top_k)

    def run_hotpot(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        return self._run_impl(question=question, answer_system_prompt=HOTPOT_SYSTEM_PROMPT, top_k=top_k)
