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
    ANSWER_SELECTION_SYSTEM_PROMPT,
    CONTROLLER_BRIDGE_PLAN_SYSTEM_PROMPT,
    CONTROLLER_DECISION_SYSTEM_PROMPT,
    CONTROLLER_GLOBAL_DECISION_SYSTEM_PROMPT,
    CONTROLLER_REWRITE_SYSTEM_PROMPT,
    HOTPOT_SYSTEM_PROMPT,
    LABEL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_bridge_plan_prompt,
    build_answer_selection_prompt,
    build_controller_prompt,
    build_global_controller_prompt,
    build_rewrite_prompt,
    build_user_prompt,
)
from src.rag.query_transform import build_retrieval_query
from src.rag.retriever import RetrievedChunk, Retriever


CONTROLLER_DECISION_MAX_TOKENS = 48
CONTROLLER_REWRITE_MAX_TOKENS = 128
CONTROLLER_TEMPERATURE = 0.0
HOTPOT_ANSWER_MAX_TOKENS = 32
ANSWER_SELECTION_MAX_TOKENS = 24


@dataclass(frozen=True)
class CorrectiveQueryPlan:
    missing_fact: str
    primary_query: str
    backup_query: str
    raw_output: str
    latency_s: float
    should_retry: bool = True


@dataclass(frozen=True)
class CorrectiveRoute:
    name: str
    reason: str
    target_doc: Optional[str] = None
    dominant_doc_share: float = 0.0


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
    answer_selection_source: Optional[str] = None
    first_pass_answer: Optional[str] = None
    corrected_pass_answer: Optional[str] = None
    answer_selection_reason: Optional[str] = None


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
    def _doc_score_table(chunks: List[RetrievedChunk]) -> dict[str, tuple[float, int]]:
        doc_scores: dict[str, tuple[float, int]] = {}
        for chunk in chunks:
            total_score, count = doc_scores.get(chunk.doc_path, (0.0, 0))
            doc_scores[chunk.doc_path] = (total_score + float(chunk.score), count + 1)
        return doc_scores

    @classmethod
    def _dominant_doc_key(cls, chunks: List[RetrievedChunk]) -> Optional[str]:
        if not chunks:
            return None
        doc_scores = cls._doc_score_table(chunks)
        ranked = sorted(doc_scores.items(), key=lambda item: (item[1][1], item[1][0]), reverse=True)
        return ranked[0][0] if ranked else None

    @classmethod
    def _dominant_doc_share(cls, chunks: List[RetrievedChunk], *, top_n: int = 5) -> float:
        top_chunks = chunks[:top_n]
        if not top_chunks:
            return 0.0
        doc_scores = cls._doc_score_table(top_chunks)
        dominant = max((count for _, count in doc_scores.values()), default=0)
        return dominant / len(top_chunks)

    def _select_corrective_route(self, question: str, chunks: List[RetrievedChunk]) -> CorrectiveRoute:
        dominant_doc = self._dominant_doc_key(chunks)
        dominant_share = self._dominant_doc_share(chunks)
        if self._is_source_specific_question(question) and dominant_doc is not None:
            if dominant_share >= 0.6:
                reason = "source_specific_question_with_dominant_document"
            else:
                reason = "source_specific_question_without_clear_clause_hit"
            return CorrectiveRoute(
                name="doc_narrow",
                reason=reason,
                target_doc=dominant_doc,
                dominant_doc_share=dominant_share,
            )
        return CorrectiveRoute(
            name="global_refine",
            reason="non_source_specific_or_no_document_anchor",
            target_doc=dominant_doc,
            dominant_doc_share=dominant_share,
        )

    @classmethod
    def _should_force_global_refine(cls, question: str, chunks: List[RetrievedChunk]) -> bool:
        if cls._is_source_specific_question(question):
            return False
        top_chunks = chunks[:5]
        if len(top_chunks) < 3:
            return False
        dominant_share = cls._dominant_doc_share(top_chunks, top_n=5)
        unique_docs = len({chunk.doc_path for chunk in top_chunks})
        # Force a second global search only when the first-pass retrieval is
        # genuinely diffuse: several sources appear, no source dominates, and
        # the top two chunks do not even agree on a likely source. This keeps
        # the trigger tied to retrieval structure rather than any benchmark.
        top_two_same_doc = top_chunks[0].doc_path == top_chunks[1].doc_path
        return dominant_share <= 0.4 and unique_docs >= 3 and not top_two_same_doc

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
        route: CorrectiveRoute,
    ) -> tuple[List[RetrievedChunk], str]:
        corrective_k = max(top_k, AGENTIC_CORRECTIVE_TOP_K)
        primary_retrieval_query, _ = self._prepare_retrieval_query(primary_query)
        if route.name == "doc_narrow" and route.target_doc:
            corrective_retrieved = self.retriever.retrieve_in_docs(
                primary_retrieval_query,
                k=corrective_k,
                doc_paths=[route.target_doc],
            )
            final_query = f"{primary_retrieval_query} @doc:{Path(route.target_doc).name}"
            fused = self._fuse_ranked_lists(previous_retrieved, corrective_retrieved)
            if backup_query != "NONE":
                backup_retrieval_query, _ = self._prepare_retrieval_query(backup_query)
                backup_retrieved = self.retriever.retrieve_in_docs(
                    backup_retrieval_query,
                    k=corrective_k,
                    doc_paths=[route.target_doc],
                )
                fused = self._fuse_ranked_lists(fused, backup_retrieved)
                final_query = (
                    f"{primary_retrieval_query} || {backup_retrieval_query} @doc:{Path(route.target_doc).name}"
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
        corrective_route: Optional[str] = None,
        route_reason: Optional[str] = None,
        route_target_doc: Optional[str] = None,
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
            "corrective_route": corrective_route,
            "route_reason": route_reason,
            "route_target_doc": route_target_doc,
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
    def _normalize_title(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    @classmethod
    def _extract_entity_candidates(cls, question: str) -> List[str]:
        candidates: List[str] = []
        for quoted in re.findall(r'"([^\"]+)"', question):
            cleaned = quoted.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
        pattern = re.compile(
            r"\b(?:[A-Z][\w'.-]*|[A-Z])(?:\s+(?:[A-Z][\w'.-]*|[A-Z]|of|the|and|or|de|la|le|vs|v|&)){0,6}"
        )
        banned = {
            "What",
            "Which",
            "Who",
            "Were",
            "Are",
            "Is",
            "The",
            "This",
            "That",
            "When",
            "Where",
            "Why",
            "How",
        }
        for match in pattern.finditer(question):
            cleaned = match.group(0).strip(" ,?.")
            if not cleaned:
                continue
            parts = re.split(r"\s+(?:and|or|vs|v|&)\s+", cleaned)
            for part in parts:
                part = part.strip(" ,?.")
                tokens = [tok for tok in part.split() if tok not in banned]
                if not tokens:
                    continue
                normalized = " ".join(tokens)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        return candidates[:8]

    @classmethod
    def _is_explicit_multi_entity_question(cls, question: str) -> bool:
        lowered = question.lower()
        entities = cls._extract_entity_candidates(question)
        if len(entities) < 2:
            return False
        comparison_cues = (
            " both ",
            " same ",
            " older ",
            " younger ",
            " higher ",
            " lower ",
            " than ",
            " versus ",
            " vs ",
        )
        if any(cue in f" {lowered} " for cue in comparison_cues):
            return True
        return bool(re.match(r"^(are|were|is|was|who is|which is)\b", lowered))

    @classmethod
    def _retrieval_covers_explicit_entities(cls, question: str, chunks: List[RetrievedChunk]) -> bool:
        if not cls._is_explicit_multi_entity_question(question):
            return False
        doc_titles = [Path(chunk.doc_path).stem for chunk in chunks[:5]]
        normalized_docs = [cls._normalize_title(title) for title in doc_titles]
        matched = 0
        for entity in cls._extract_entity_candidates(question):
            normalized_entity = cls._normalize_title(entity)
            if not normalized_entity:
                continue
            if any(
                normalized_entity == doc
                or normalized_entity in doc
                or doc in normalized_entity
                for doc in normalized_docs
            ):
                matched += 1
        return matched >= 2


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
            should_retry=True,
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
            should_retry=True,
        )

    def _parse_bridge_plan_output(self, text: str, *, question: str) -> CorrectiveQueryPlan:
        plan = self._parse_rewrite_output(text, question=question)
        cleaned = text.strip()
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
            return plan
        need_retry = str(payload.get("need_retry", "")).strip().upper()
        if need_retry == "NO":
            return CorrectiveQueryPlan(
                missing_fact=plan.missing_fact,
                primary_query="NONE",
                backup_query="NONE",
                raw_output=text,
                latency_s=0.0,
                should_retry=False,
            )
        return plan

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

    def _decision_step(self, question: str, context: str, route: CorrectiveRoute) -> tuple[str, float, str]:
        if route.name == "doc_narrow":
            user_prompt = build_controller_prompt(context=context, question=question)
            system_prompt = CONTROLLER_DECISION_SYSTEM_PROMPT
        else:
            user_prompt = build_global_controller_prompt(context=context, question=question)
            system_prompt = CONTROLLER_GLOBAL_DECISION_SYSTEM_PROMPT
        started = perf_counter()
        result = self.model.generate(
            system_prompt,
            user_prompt,
            temperature=CONTROLLER_TEMPERATURE,
            max_tokens=CONTROLLER_DECISION_MAX_TOKENS,
        )
        latency_s = perf_counter() - started
        decision = self._parse_decision_output(result)
        return decision, latency_s, result

    def _rewrite_step(self, question: str, context: str, route: CorrectiveRoute) -> CorrectiveQueryPlan:
        entity_hints = self._entity_hints(question)
        if route.name == "doc_narrow":
            user_prompt = build_rewrite_prompt(
                question=question,
                entity_hints=entity_hints,
                context=context,
            )
            system_prompt = CONTROLLER_REWRITE_SYSTEM_PROMPT
        else:
            user_prompt = build_bridge_plan_prompt(
                question=question,
                entity_hints=entity_hints,
                context=context,
            )
            system_prompt = CONTROLLER_BRIDGE_PLAN_SYSTEM_PROMPT
        started = perf_counter()
        result = self.model.generate(
            system_prompt,
            user_prompt,
            temperature=CONTROLLER_TEMPERATURE,
            max_tokens=CONTROLLER_REWRITE_MAX_TOKENS,
        )
        latency_s = perf_counter() - started
        if route.name == "doc_narrow":
            plan = self._parse_rewrite_output(result, question=question)
        else:
            plan = self._parse_bridge_plan_output(result, question=question)
        return CorrectiveQueryPlan(
            missing_fact=plan.missing_fact,
            primary_query=plan.primary_query,
            backup_query=plan.backup_query,
            raw_output=result,
            latency_s=latency_s,
            should_retry=plan.should_retry,
        )

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _parse_answer_selection_output(self, text: str) -> tuple[str, str]:
        cleaned = text.strip()
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
            return "B", "selector_parse_fallback"
        choice = str(payload.get("choice", "")).strip().upper()
        reason = self._normalize_optional_text(payload.get("reason"))
        if choice in {"A", "B"}:
            return choice, reason
        return "UNKNOWN", reason

    def _select_hotpot_answer(
        self,
        *,
        question: str,
        first_context: str,
        first_answer: str,
        corrected_context: str,
        corrected_answer: str,
    ) -> tuple[str, str, str, float]:
        first_norm = self._normalize_answer_text(first_answer)
        corrected_norm = self._normalize_answer_text(corrected_answer)
        if first_norm == corrected_norm:
            return corrected_answer, "corrected_pass", "same_answer", 0.0
        if corrected_norm == "unknown" and first_norm != "unknown":
            return first_answer, "first_pass", "fallback_from_unknown", 0.0
        if first_norm == "unknown" and corrected_norm != "unknown":
            return corrected_answer, "corrected_pass", "improved_over_unknown", 0.0

        selector_prompt = build_answer_selection_prompt(
            question=question,
            answer_a=first_answer,
            context_a=first_context,
            answer_b=corrected_answer,
            context_b=corrected_context,
        )
        started = perf_counter()
        raw = self.model.generate(
            ANSWER_SELECTION_SYSTEM_PROMPT,
            selector_prompt,
            temperature=CONTROLLER_TEMPERATURE,
            max_tokens=ANSWER_SELECTION_MAX_TOKENS,
        )
        latency_s = perf_counter() - started
        choice, reason = self._parse_answer_selection_output(raw)
        if choice == "A":
            return first_answer, "first_pass", reason, latency_s
        if choice == "UNKNOWN":
            return "unknown", "selector_unknown", reason, latency_s
        return corrected_answer, "corrected_pass", reason, latency_s

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
        first_pass_context = ""
        first_pass_used: List[RetrievedChunk] = []
        first_pass_retrieved: List[RetrievedChunk] = []

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
            context, used = build_context(
                retrieved,
                char_budget=CONTEXT_CHAR_BUDGET,
            )
            if i == 0:
                first_pass_context = context
                first_pass_used = list(used)
                first_pass_retrieved = list(retrieved)
            current_signature = self._chunk_signature(retrieved)
            controller_context = self._controller_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
            route = self._select_corrective_route(question, retrieved)
            plan: Optional[CorrectiveQueryPlan] = None

            if route.name == "doc_narrow":
                decision, controller_latency_s, raw_controller_output = self._decision_step(
                    question, controller_context, route
                )
                generation_latency_s += controller_latency_s
            else:
                if self._retrieval_covers_explicit_entities(question, retrieved):
                    plan = CorrectiveQueryPlan(
                        missing_fact="NONE",
                        primary_query="NONE",
                        backup_query="NONE",
                        raw_output="AUTO_STOP>> first-pass retrieval already covers the explicit entity pages",
                        latency_s=0.0,
                        should_retry=False,
                    )
                    controller_latency_s = 0.0
                    raw_controller_output = plan.raw_output
                    decision = "STOP"
                else:
                    plan = self._rewrite_step(question, controller_context, route)
                    generation_latency_s += plan.latency_s
                    controller_latency_s = plan.latency_s
                    raw_controller_output = f"GLOBAL_PLAN>> {plan.raw_output}"
                    decision = "RETRY" if plan.should_retry else "STOP"

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
                            missing_aspect=(plan.missing_fact if plan is not None else "NONE"),
                            raw_controller_output=raw_controller_output,
                            corrective_action="none",
                            corrective_route=route.name,
                            route_reason=("global_plan_stop" if route.name != "doc_narrow" else "controller_stop"),
                            route_target_doc=route.target_doc,
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
                                corrective_route=route.name,
                                route_reason=route.reason,
                                route_target_doc=route.target_doc,
                            )
                        )
                    break
                if plan is None:
                    plan = self._rewrite_step(question, controller_context, route)
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
                                stop_reason=("global_plan_no_retry" if not plan.should_retry else "empty_refinement"),
                                controller_decision=decision,
                                controller_confidence=None,
                                missing_aspect=plan.missing_fact,
                                raw_controller_output=(
                                    raw_controller_output
                                    if route.name != "doc_narrow"
                                    else f"DECISION>> {raw_controller_output}\nREWRITE>> {plan.raw_output}"
                                ),
                                corrective_action="none",
                                corrective_route=route.name,
                                route_reason=route.reason,
                                route_target_doc=route.target_doc,
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
                                corrective_route=route.name,
                                route_reason=route.reason,
                                route_target_doc=route.target_doc,
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
                    route=route,
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
                                corrective_route=route.name,
                                route_reason=route.reason,
                                route_target_doc=route.target_doc,
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
                            corrective_route=route.name,
                            route_reason=route.reason,
                            route_target_doc=route.target_doc,
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

        final_retrieved = best_stop_retrieved or retrieved
        final_context, final_used = build_context(
            final_retrieved,
            char_budget=CONTEXT_CHAR_BUDGET,
        )

        user_prompt = build_user_prompt(context=final_context, question=question)
        generation_started = perf_counter()
        final_system_prompt = answer_system_prompt
        if not best_stop_context:
            final_system_prompt = (
                answer_system_prompt
                + " The retrieval controller did not confirm sufficient evidence. "
                + "If the context does not directly support the answer, say you don't know."
            )
        if answer_system_prompt == HOTPOT_SYSTEM_PROMPT:
            answer = self.model.generate(
                final_system_prompt,
                user_prompt,
                max_tokens=HOTPOT_ANSWER_MAX_TOKENS,
            )
        else:
            answer = self.model.generate(final_system_prompt, user_prompt)
        generation_latency_s += perf_counter() - generation_started

        answer_selection_source: Optional[str] = "single_pass"
        answer_selection_reason: Optional[str] = "single_pass_only"
        first_pass_answer: Optional[str] = None
        corrected_pass_answer: Optional[str] = None

        if answer_system_prompt == HOTPOT_SYSTEM_PROMPT and corrective_pass_used and first_pass_context:
            first_user_prompt = build_user_prompt(context=first_pass_context, question=question)
            first_started = perf_counter()
            first_candidate_answer = self.model.generate(
                HOTPOT_SYSTEM_PROMPT,
                first_user_prompt,
                max_tokens=HOTPOT_ANSWER_MAX_TOKENS,
            )
            generation_latency_s += perf_counter() - first_started

            corrected_pass_answer = answer
            first_pass_answer = first_candidate_answer

            selected_answer, answer_selection_source, answer_selection_reason, selection_latency_s = (
                self._select_hotpot_answer(
                    question=question,
                    first_context=first_pass_context,
                    first_answer=first_candidate_answer,
                    corrected_context=final_context,
                    corrected_answer=answer,
                )
            )
            generation_latency_s += selection_latency_s
            answer = selected_answer
            if answer_selection_source == "first_pass":
                final_retrieved = first_pass_retrieved
                final_used = first_pass_used
                final_context = first_pass_context
                user_prompt = first_user_prompt
                corrected_pass_answer = corrected_pass_answer
            else:
                corrected_pass_answer = corrected_pass_answer

        final_system_prompt_for_count = HOTPOT_SYSTEM_PROMPT if answer_system_prompt == HOTPOT_SYSTEM_PROMPT else final_system_prompt

        return AgenticResult(
            answer=answer,
            retrieved=final_retrieved,
            used=final_used,
            iterations=len(refined_queries) + 1,
            refined_queries=refined_queries,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(final_context),
            prompt_tokens=self.model.count_tokens(final_system_prompt_for_count) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
            iteration_trace=iteration_trace,
            answer_selection_source=answer_selection_source,
            first_pass_answer=first_pass_answer,
            corrected_pass_answer=corrected_pass_answer,
            answer_selection_reason=answer_selection_reason,
        )

    def run(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        return self._run_impl(question=question, answer_system_prompt=SYSTEM_PROMPT, top_k=top_k)

    def run_label(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        return self._run_impl(question=question, answer_system_prompt=LABEL_SYSTEM_PROMPT, top_k=top_k)

    def run_hotpot(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        return self._run_impl(question=question, answer_system_prompt=HOTPOT_SYSTEM_PROMPT, top_k=top_k)
