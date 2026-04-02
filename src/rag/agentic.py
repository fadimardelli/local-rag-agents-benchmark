from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from typing import List, Optional

from src.config.defaults import (
    AGENTIC_MAX_ITERS,
    AGENTIC_CORRECTIVE_RRF_K,
    AGENTIC_CORRECTIVE_TOP_K,
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
CONTROLLER_REWRITE_MAX_TOKENS = 128
CONTROLLER_TEMPERATURE = 0.0


@dataclass(frozen=True)
class CorrectiveQueryPlan:
    missing_fact: str
    primary_query: str
    backup_query: str
    raw_output: str
    latency_s: float


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
        query_transform_mode: Optional[str] = None,
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
        self.query_transform_mode = query_transform_mode or QUERY_TRANSFORM_MODE

    @staticmethod
    def _is_source_specific_question(question: str) -> bool:
        lowered = question.lower()
        return ";" in question or "agreement" in lowered or "document" in lowered or "policy" in lowered

    @staticmethod
    def _dominant_doc_key(chunks: List[RetrievedChunk]) -> Optional[str]:
        if not chunks:
            return None
        doc_scores: dict[str, tuple[float, int]] = {}
        for chunk in chunks:
            total_score, count = doc_scores.get(chunk.doc_path, (0.0, 0))
            doc_scores[chunk.doc_path] = (total_score + float(chunk.score), count + 1)
        ranked = sorted(doc_scores.items(), key=lambda item: (item[1][1], item[1][0]), reverse=True)
        return ranked[0][0] if ranked else None

    def _apply_source_focus(self, question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not self._is_source_specific_question(question):
            return chunks
        dominant_doc = self._dominant_doc_key(chunks)
        if dominant_doc is None:
            return chunks
        focused = [chunk for chunk in chunks if chunk.doc_path == dominant_doc]
        return focused or chunks

    @staticmethod
    def _fuse_ranked_lists(
        primary: List[RetrievedChunk],
        secondary: List[RetrievedChunk],
        *,
        rrf_k: int = AGENTIC_CORRECTIVE_RRF_K,
    ) -> List[RetrievedChunk]:
        fused_scores: dict[tuple[str, int, int], float] = {}
        fused_chunks: dict[tuple[str, int, int], RetrievedChunk] = {}
        for ranked in (primary, secondary):
            for rank, chunk in enumerate(ranked, start=1):
                key = (chunk.doc_path, chunk.start, chunk.end)
                fused_chunks[key] = chunk
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
        return [
            RetrievedChunk(
                doc_path=fused_chunks[key].doc_path,
                start=fused_chunks[key].start,
                end=fused_chunks[key].end,
                text=fused_chunks[key].text,
                score=fused_scores[key],
            )
            for key in ranked_keys
        ]

    def _corrective_retrieve(
        self,
        *,
        primary_query: str,
        backup_query: str,
        previous_retrieved: List[RetrievedChunk],
        top_k: int,
        question: str,
    ) -> tuple[List[RetrievedChunk], str]:
        corrective_k = max(top_k, AGENTIC_CORRECTIVE_TOP_K)
        primary_retrieval_query, _ = self._prepare_retrieval_query(primary_query)
        target_doc = self._dominant_doc_key(previous_retrieved)
        if target_doc and self._is_source_specific_question(question):
            corrective_retrieved = self.retriever.retrieve_in_docs(
                primary_retrieval_query,
                k=corrective_k,
                doc_paths=[target_doc],
            )
            final_query = f"{primary_retrieval_query} @doc:{Path(target_doc).name}"
            fused = self._fuse_ranked_lists(previous_retrieved, corrective_retrieved)
            if backup_query != "NONE":
                backup_retrieval_query, _ = self._prepare_retrieval_query(backup_query)
                backup_retrieved = self.retriever.retrieve_in_docs(
                    backup_retrieval_query,
                    k=corrective_k,
                    doc_paths=[target_doc],
                )
                fused = self._fuse_ranked_lists(fused, backup_retrieved)
                final_query = (
                    f"{primary_retrieval_query} || {backup_retrieval_query} @doc:{Path(target_doc).name}"
                )
            return fused, final_query

        corrective_retrieved = self.retriever.retrieve(primary_retrieval_query, k=corrective_k)
        fused = self._fuse_ranked_lists(previous_retrieved, corrective_retrieved)
        final_query = primary_retrieval_query
        if backup_query != "NONE":
            backup_retrieval_query, _ = self._prepare_retrieval_query(backup_query)
            backup_retrieved = self.retriever.retrieve(backup_retrieval_query, k=corrective_k)
            fused = self._fuse_ranked_lists(fused, backup_retrieved)
            final_query = f"{primary_retrieval_query} || {backup_retrieval_query}"
        return self._apply_source_focus(question, fused), final_query

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
        corrective_action: Optional[str] = None,
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
            "corrective_action": corrective_action,
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

    @classmethod
    def _normalize_query_candidate(cls, text: Optional[str]) -> str:
        cleaned = cls._normalize_optional_text(text)
        if cleaned == "NONE":
            return "NONE"
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()
        if any(token in cleaned for token in ("AND", " OR ", "(", ")", '"', "'")):
            return "NONE"
        return cls._normalize_optional_text(cleaned)

    def _parse_rewrite_output(self, text: str, *, question: str) -> CorrectiveQueryPlan:
        fallback = CorrectiveQueryPlan(
            missing_fact="NONE",
            primary_query="NONE",
            backup_query="NONE",
            raw_output=text,
            latency_s=0.0,
        )
        cleaned = text.strip()
        if not cleaned:
            return fallback
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()
        if not cleaned.startswith("{"):
            match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
            if match:
                cleaned = match.group(0).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            return fallback

        missing_fact = self._normalize_optional_text(payload.get("missing_fact"))
        primary_query = self._normalize_query_candidate(payload.get("primary_query"))
        backup_query = self._normalize_query_candidate(payload.get("backup_query"))

        original_norm = self._normalize_query(question)
        if primary_query != "NONE" and self._normalize_query(primary_query) == original_norm:
            primary_query = "NONE"
        if backup_query != "NONE" and self._normalize_query(backup_query) == original_norm:
            backup_query = "NONE"
        if backup_query != "NONE" and primary_query != "NONE":
            if self._normalize_query(backup_query) == self._normalize_query(primary_query):
                backup_query = "NONE"

        return CorrectiveQueryPlan(
            missing_fact=missing_fact,
            primary_query=primary_query,
            backup_query=backup_query,
            raw_output=text,
            latency_s=0.0,
        )

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

    def _rewrite_step(self, question: str, context: str) -> CorrectiveQueryPlan:
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
        plan = self._parse_rewrite_output(result, question=question)
        return CorrectiveQueryPlan(
            missing_fact=plan.missing_fact,
            primary_query=plan.primary_query,
            backup_query=plan.backup_query,
            raw_output=result,
            latency_s=latency_s,
        )

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
        pending_retrieved: Optional[List[RetrievedChunk]] = None
        pending_retrieval_query: Optional[str] = None
        corrective_pass_used = False
        best_stop_retrieved: List[RetrievedChunk] = []
        best_stop_used: List[RetrievedChunk] = []
        best_stop_context = ""

        for i in range(AGENTIC_MAX_ITERS):
            if pending_retrieved is not None and pending_retrieval_query is not None:
                retrieved = pending_retrieved
                retrieval_query = pending_retrieval_query
                pending_retrieved = None
                pending_retrieval_query = None
            else:
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
                if corrective_pass_used:
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
                                is_sufficient=False,
                                assess_latency_s=controller_latency_s,
                                stop_reason="max_corrective_passes",
                                controller_decision=decision,
                                controller_confidence=None,
                                missing_aspect="NONE",
                                raw_controller_output=raw_controller_output,
                                corrective_action="none",
                            )
                        )
                    break
                plan = self._rewrite_step(question, controller_context)
                generation_latency_s += plan.latency_s
                if plan.primary_query == "NONE":
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
                                refine_latency_s=plan.latency_s,
                                refined_query=plan.primary_query,
                                stop_reason="empty_refinement",
                                controller_decision=decision,
                                controller_confidence=None,
                                missing_aspect=plan.missing_fact,
                                raw_controller_output=f"DECISION>> {raw_controller_output}\nREWRITE>> {plan.raw_output}",
                                corrective_action="none",
                            )
                        )
                    break
                if self._normalize_query(plan.primary_query) == self._normalize_query(current_query):
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
                                refine_latency_s=plan.latency_s,
                                refined_query=plan.primary_query,
                                stop_reason="unchanged_refinement",
                                controller_decision=decision,
                                controller_confidence=None,
                                missing_aspect=plan.missing_fact,
                                raw_controller_output=f"DECISION>> {raw_controller_output}\nREWRITE>> {plan.raw_output}",
                                corrective_action="none",
                            )
                        )
                    break
                corrective_started = perf_counter()
                corrected_retrieved, corrective_retrieval_query = self._corrective_retrieve(
                    primary_query=plan.primary_query,
                    backup_query=plan.backup_query,
                    previous_retrieved=retrieved,
                    top_k=top_k,
                    question=question,
                )
                retrieval_latency_s += perf_counter() - corrective_started
                corrected_signature = self._chunk_signature(corrected_retrieved)

                if previous_signature is not None and corrected_signature == previous_signature:
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
                                refine_latency_s=plan.latency_s,
                                refined_query=plan.primary_query,
                                stop_reason="repeated_retrieval",
                                controller_decision=decision,
                                controller_confidence=None,
                                missing_aspect=plan.missing_fact,
                                raw_controller_output=f"DECISION>> {raw_controller_output}\nREWRITE>> {plan.raw_output}",
                                corrective_action="repeat_after_corrective_retrieval",
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
                            refine_latency_s=plan.latency_s,
                            refined_query=plan.primary_query,
                            stop_reason="continue",
                            controller_decision=decision,
                            controller_confidence=None,
                            missing_aspect=plan.missing_fact,
                            raw_controller_output=f"DECISION>> {raw_controller_output}\nREWRITE>> {plan.raw_output}",
                            corrective_action="rewrite_plus_corrective_retrieval",
                        )
                    )
                refined_queries.append(plan.primary_query)
                corrective_pass_used = True
                current_query = plan.primary_query
                previous_signature = corrected_signature
                pending_retrieved = corrected_retrieved
                pending_retrieval_query = corrective_retrieval_query
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
                        corrective_action="none",
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
