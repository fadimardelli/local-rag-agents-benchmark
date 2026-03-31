from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Dict, List, Optional, Set

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
    HOTPOT_SYSTEM_PROMPT,
    LABEL_SYSTEM_PROMPT,
    MULTI_ACTION_CONTROLLER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_multi_action_controller_prompt,
    build_user_prompt,
)
from src.rag.query_transform import build_retrieval_query
from src.rag.retriever import RetrievedChunk, Retriever


CONTROLLER_MAX_TOKENS = 160
CONTROLLER_TEMPERATURE = 0.0
ACTION_SEARCH_CHUNKS = "SEARCH_CHUNKS"
ACTION_SELECT_DOCUMENT = "SELECT_DOCUMENT"
ACTION_REFORMULATE_QUERY = "REFORMULATE_QUERY"
ACTION_EXPAND_DOCUMENT = "EXPAND_DOCUMENT"
ACTION_STOP = "STOP"
VALID_ACTIONS = {
    ACTION_SEARCH_CHUNKS,
    ACTION_SELECT_DOCUMENT,
    ACTION_REFORMULATE_QUERY,
    ACTION_EXPAND_DOCUMENT,
    ACTION_STOP,
}

CONTRACTNLI_DOC_KNOWN_ACTIONS = {
    ACTION_SEARCH_CHUNKS,
    ACTION_REFORMULATE_QUERY,
    ACTION_EXPAND_DOCUMENT,
    ACTION_STOP,
}


@dataclass(frozen=True)
class MultiActionDecision:
    action: str
    query: str
    document_index: Optional[int]
    anchor_chunk_index: Optional[int]
    rationale: str


@dataclass(frozen=True)
class SourceEvidence:
    score: float
    matched_tokens: List[str]


@dataclass(frozen=True)
class MultiActionAgenticResult:
    answer: str
    retrieved: List[RetrievedChunk]
    used: List[RetrievedChunk]
    iterations: int
    retrieval_steps: int
    action_history: List[str]
    refined_queries: List[str]
    retrieval_latency_s: float
    generation_latency_s: float
    retrieved_context_tokens: int
    prompt_tokens: int
    answer_tokens: int
    iteration_trace: List[dict]


class MultiActionAgenticRAG:
    SOURCE_STOPWORDS = {
        "the",
        "and",
        "agreement",
        "document",
        "party",
        "parties",
        "confidential",
        "information",
        "receiving",
        "disclosing",
        "does",
        "grant",
        "rights",
        "right",
        "indicate",
        "between",
        "under",
        "this",
        "that",
        "with",
        "from",
        "for",
        "into",
        "any",
        "not",
        "non",
        "disclosure",
        "nda",
        "contract",
        "question",
        "consider",
    }
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

    @staticmethod
    def _normalize_query(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _normalize_optional_text(text: Optional[str]) -> str:
        if text is None:
            return ""
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return ""
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()
        return cleaned

    @staticmethod
    def _chunk_signature(chunks: List[RetrievedChunk]) -> tuple[tuple[str, int, int], ...]:
        return tuple((chunk.doc_path, chunk.start, chunk.end) for chunk in chunks)

    @staticmethod
    def _history_text(history: List[str]) -> str:
        return "\n".join(f"{idx}. {entry}" for idx, entry in enumerate(history, start=1)) or "NONE"

    @staticmethod
    def _focused_document_text(doc_path: Optional[str]) -> str:
        return Path(doc_path).name if doc_path else "NONE"

    @classmethod
    def _tokenize_source_text(cls, text: str) -> List[str]:
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
        text = re.sub(r"\b\d+(?:\.\d+)*\b", " ", text)
        tokens: List[str] = []
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'.&]*", text):
            cleaned = re.sub(r"[^A-Za-z0-9]+", "", token).lower()
            if len(cleaned) >= 3 and cleaned not in cls.SOURCE_STOPWORDS:
                tokens.append(cleaned)
        return tokens

    @classmethod
    def _salient_query_tokens(cls, question: str) -> List[str]:
        tokens: List[str] = []

        between_match = re.search(r"\bbetween\s+(.+?);", question, flags=re.IGNORECASE)
        if between_match:
            tokens.extend(cls._tokenize_source_text(between_match.group(1)))

        for token in re.findall(r"\b(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Za-z]*[A-Z][A-Za-z]*)\b", question):
            tokens.extend(cls._tokenize_source_text(token))

        deduped: List[str] = []
        for token in tokens:
            if token not in deduped:
                deduped.append(token)
        return deduped

    @classmethod
    def _doc_title_tokens(cls, doc_path: str) -> List[str]:
        name = Path(doc_path).name
        stem = name.replace(".txt", " ").replace(".pdf", " ").replace("||", " ")
        return cls._tokenize_source_text(stem)

    @classmethod
    def _source_evidence(cls, question: str, doc_path: str) -> SourceEvidence:
        query_tokens = cls._salient_query_tokens(question)
        if not query_tokens:
            return SourceEvidence(score=0.0, matched_tokens=[])

        doc_tokens = set(cls._doc_title_tokens(doc_path))
        matched = [token for token in query_tokens if token in doc_tokens]
        return SourceEvidence(
            score=len(matched) / max(1, len(query_tokens)),
            matched_tokens=matched,
        )

    @classmethod
    def _controller_context(cls, question: str, chunks: List[RetrievedChunk], char_budget: int) -> str:
        parts: List[str] = []
        total = 0
        document_ids: Dict[str, int] = {}
        for chunk in chunks:
            if chunk.doc_path not in document_ids:
                document_ids[chunk.doc_path] = len(document_ids) + 1

        for doc_path, doc_id in document_ids.items():
            source = Path(doc_path).name
            source_evidence = cls._source_evidence(question, doc_path)
            matched = ", ".join(source_evidence.matched_tokens) if source_evidence.matched_tokens else "NONE"
            doc_chunks = [chunk for chunk in chunks if chunk.doc_path == doc_path]
            top_span = doc_chunks[0] if doc_chunks else None
            block = (
                f"[Document {doc_id}]\n"
                f"Source: {source}\n"
                f"SourceMatchScore: {source_evidence.score:.2f}\n"
                f"SourceMatchTokens: {matched}\n"
                f"RetrievedChunkCount: {len(doc_chunks)}\n"
            )
            if top_span is not None:
                block += f"TopRetrievedSpan: {top_span.start}-{top_span.end}\n"
            add_len = len(block) + (2 if parts else 0)
            if total + add_len > char_budget:
                break
            if parts:
                parts.append("\n\n")
            parts.append(block)
            total += add_len

        for idx, chunk in enumerate(chunks, start=1):
            source = Path(chunk.doc_path).name
            source_evidence = cls._source_evidence(question, chunk.doc_path)
            matched = ", ".join(source_evidence.matched_tokens) if source_evidence.matched_tokens else "NONE"
            block = (
                f"[Chunk {idx}]\n"
                f"DocumentId: {document_ids[chunk.doc_path]}\n"
                f"Source: {source}\n"
                f"SourceMatchScore: {source_evidence.score:.2f}\n"
                f"SourceMatchTokens: {matched}\n"
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

    @staticmethod
    def _fallback_action(text: str) -> str:
        upper = text.strip().upper()
        for action in VALID_ACTIONS:
            if action in upper:
                return action
        first_word = upper.split()[0] if upper else ""
        return first_word if first_word in VALID_ACTIONS else ACTION_STOP

    @staticmethod
    def _parse_chunk_index(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 1 else None

    def _parse_controller_output(self, text: str, current_query: str) -> MultiActionDecision:
        stripped = text.strip()
        if not stripped:
            return MultiActionDecision(
                action=ACTION_STOP,
                query=current_query,
                document_index=None,
                anchor_chunk_index=None,
                rationale="",
            )

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None
            else:
                payload = None

        if isinstance(payload, dict):
            raw_action = payload.get("action")
            action = self._normalize_optional_text(raw_action if isinstance(raw_action, str) else None).upper()
            if action not in VALID_ACTIONS:
                action = self._fallback_action(stripped)
            raw_query = payload.get("query")
            raw_rationale = payload.get("rationale")
            query = self._normalize_optional_text(raw_query if isinstance(raw_query, str) else None) or current_query
            rationale = self._normalize_optional_text(raw_rationale if isinstance(raw_rationale, str) else None)
            document_index = self._parse_chunk_index(payload.get("document_id"))
            anchor_chunk_index = self._parse_chunk_index(payload.get("anchor_chunk"))
            return MultiActionDecision(
                action=action,
                query=query,
                document_index=document_index,
                anchor_chunk_index=anchor_chunk_index,
                rationale=rationale,
            )

        action = self._fallback_action(stripped)
        return MultiActionDecision(
            action=action,
            query=current_query,
            document_index=None,
            anchor_chunk_index=None,
            rationale="",
        )

    def _make_iteration_entry(
        self,
        *,
        iteration_index: int,
        action: str,
        controller_query: str,
        controller_retrieval_query: str,
        controller_retrieved: List[RetrievedChunk],
        controller_used: List[RetrievedChunk],
        controller_context: str,
        stop_reason: Optional[str],
        next_query: Optional[str],
        document_index: Optional[int],
        anchor_chunk_index: Optional[int],
        controller_latency_s: float,
        raw_controller_output: str,
        rationale: str,
        result_query: str,
        result_retrieval_query: str,
        focused_document: Optional[str],
        result_focused_document: Optional[str],
        result_retrieved: List[RetrievedChunk],
        result_used: List[RetrievedChunk],
        result_context: str,
        changed_retrieval: bool,
        source_guard_applied: bool,
        anchor_source_score: float,
        best_source_score: float,
        stop_guard_applied: bool,
    ) -> dict:
        return {
            "iteration_index": iteration_index,
            "action": action,
            "controller_query": controller_query,
            "controller_retrieval_query": controller_retrieval_query,
            "controller_retrieved_refs": [self._chunk_ref(chunk) for chunk in controller_retrieved],
            "controller_used_refs": [self._chunk_ref(chunk) for chunk in controller_used],
            "controller_context_chars": len(controller_context),
            "stop_reason": stop_reason,
            "next_query": next_query,
            "document_index": document_index,
            "anchor_chunk_index": anchor_chunk_index,
            "controller_latency_s": controller_latency_s,
            "raw_controller_output": raw_controller_output,
            "rationale": rationale,
            "result_query": result_query,
            "result_retrieval_query": result_retrieval_query,
            "focused_document": focused_document,
            "result_focused_document": result_focused_document,
            "result_retrieved_refs": [self._chunk_ref(chunk) for chunk in result_retrieved],
            "result_used_refs": [self._chunk_ref(chunk) for chunk in result_used],
            "result_context_chars": len(result_context),
            "changed_retrieval": changed_retrieval,
            "source_guard_applied": source_guard_applied,
            "anchor_source_score": anchor_source_score,
            "best_source_score": best_source_score,
            "stop_guard_applied": stop_guard_applied,
        }

    def _controller_step(
        self,
        *,
        question: str,
        current_query: str,
        focused_document: Optional[str],
        available_actions: Set[str],
        action_history: List[str],
        retrieved: List[RetrievedChunk],
    ) -> tuple[MultiActionDecision, float, str]:
        context = self._controller_context(question, retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        user_prompt = build_multi_action_controller_prompt(
            question=question,
            current_query=current_query,
            focused_document=self._focused_document_text(focused_document),
            available_actions=", ".join(sorted(available_actions)),
            action_history=self._history_text(action_history),
            retrieved_context=context,
        )
        started = perf_counter()
        raw_output = self.model.generate(
            MULTI_ACTION_CONTROLLER_SYSTEM_PROMPT,
            user_prompt,
            temperature=CONTROLLER_TEMPERATURE,
            max_tokens=CONTROLLER_MAX_TOKENS,
        )
        latency_s = perf_counter() - started
        decision = self._parse_controller_output(raw_output, current_query)
        if decision.action not in available_actions:
            fallback_action = ACTION_EXPAND_DOCUMENT if ACTION_EXPAND_DOCUMENT in available_actions else ACTION_SEARCH_CHUNKS
            decision = MultiActionDecision(
                action=fallback_action,
                query=decision.query,
                document_index=None,
                anchor_chunk_index=decision.anchor_chunk_index,
                rationale=(
                    f"{decision.rationale} Requested action was not available in this task setting."
                ).strip(),
            )
        return decision, latency_s, raw_output

    @staticmethod
    def _resolve_anchor_chunk(retrieved: List[RetrievedChunk], anchor_chunk_index: Optional[int]) -> Optional[RetrievedChunk]:
        if not retrieved:
            return None
        if anchor_chunk_index is None:
            return retrieved[0]
        zero_based = anchor_chunk_index - 1
        if zero_based < 0 or zero_based >= len(retrieved):
            return retrieved[0]
        return retrieved[zero_based]

    @staticmethod
    def _resolve_document_path(retrieved: List[RetrievedChunk], document_index: Optional[int]) -> Optional[str]:
        if not retrieved or document_index is None or document_index < 1:
            return None
        ordered_docs: List[str] = []
        for chunk in retrieved:
            if chunk.doc_path not in ordered_docs:
                ordered_docs.append(chunk.doc_path)
        zero_based = document_index - 1
        if zero_based >= len(ordered_docs):
            return None
        return ordered_docs[zero_based]

    @classmethod
    def _anchor_is_source_plausible(
        cls,
        *,
        question: str,
        retrieved: List[RetrievedChunk],
        anchor_chunk_index: Optional[int],
    ) -> tuple[bool, float, float]:
        if not retrieved:
            return True, 0.0, 0.0

        if not cls._salient_query_tokens(question):
            return True, 0.0, 0.0

        anchor = cls._resolve_anchor_chunk(retrieved, anchor_chunk_index)
        if anchor is None:
            return True, 0.0, 0.0

        anchor_score = cls._source_evidence(question, anchor.doc_path).score
        best_score = max(cls._source_evidence(question, chunk.doc_path).score for chunk in retrieved)
        is_plausible = anchor_score >= 0.5 * best_score and (anchor_score > 0.0 or best_score == 0.0)
        return is_plausible, anchor_score, best_score

    @classmethod
    def _apply_source_guard(
        cls,
        *,
        question: str,
        retrieved: List[RetrievedChunk],
        decision: MultiActionDecision,
        locked_document_path: Optional[str] = None,
    ) -> tuple[MultiActionDecision, bool, float, float]:
        if decision.action not in {ACTION_SELECT_DOCUMENT, ACTION_EXPAND_DOCUMENT}:
            return decision, False, 0.0, 0.0

        if locked_document_path:
            return decision, False, 1.0, 1.0

        if decision.action == ACTION_SELECT_DOCUMENT:
            doc_path = cls._resolve_document_path(retrieved, decision.document_index)
            if doc_path is None:
                return decision, False, 0.0, 0.0
            anchor_score = cls._source_evidence(question, doc_path).score
            best_score = max(cls._source_evidence(question, chunk.doc_path).score for chunk in retrieved)
            is_plausible = (
                anchor_score >= 0.5 * best_score and (anchor_score > 0.0 or best_score == 0.0)
                and cls._document_commitment_is_justified(
                    question=question,
                    retrieved=retrieved,
                    doc_path=doc_path,
                )
            )
        else:
            is_plausible, anchor_score, best_score = cls._anchor_is_source_plausible(
                question=question,
                retrieved=retrieved,
                anchor_chunk_index=decision.anchor_chunk_index,
            )
            anchor = cls._resolve_anchor_chunk(retrieved, decision.anchor_chunk_index)
            is_plausible = is_plausible and cls._document_commitment_is_justified(
                question=question,
                retrieved=retrieved,
                doc_path=anchor.doc_path if anchor is not None else None,
            )
        if is_plausible:
            return decision, False, anchor_score, best_score

        guarded_decision = MultiActionDecision(
            action=ACTION_SEARCH_CHUNKS,
            query=decision.query,
            document_index=None,
            anchor_chunk_index=None,
            rationale=(
                f"{decision.rationale} "
                f"Document-local exploration was deferred because no single retrieved document is yet sufficiently justified."
            ).strip(),
        )
        return guarded_decision, True, anchor_score, best_score

    @classmethod
    def _dominant_doc_stats(cls, retrieved: List[RetrievedChunk]) -> tuple[Optional[str], float]:
        if not retrieved:
            return None, 0.0
        counts: Dict[str, int] = {}
        for chunk in retrieved:
            counts[chunk.doc_path] = counts.get(chunk.doc_path, 0) + 1
        dominant_doc, dominant_count = max(counts.items(), key=lambda item: item[1])
        return dominant_doc, dominant_count / len(retrieved)

    @staticmethod
    def _document_support_stats(retrieved: List[RetrievedChunk]) -> tuple[Dict[str, int], Dict[str, float]]:
        counts: Dict[str, int] = {}
        score_sums: Dict[str, float] = {}
        for chunk in retrieved:
            counts[chunk.doc_path] = counts.get(chunk.doc_path, 0) + 1
            score_sums[chunk.doc_path] = score_sums.get(chunk.doc_path, 0.0) + float(chunk.score)
        return counts, score_sums

    @classmethod
    def _document_commitment_is_justified(
        cls,
        *,
        question: str,
        retrieved: List[RetrievedChunk],
        doc_path: Optional[str],
    ) -> bool:
        if not retrieved or not doc_path:
            return False

        if cls._source_evidence(question, doc_path).score > 0.0:
            return True

        counts, score_sums = cls._document_support_stats(retrieved)
        doc_count = counts.get(doc_path, 0)
        if doc_count == 0:
            return False

        ranked_docs = sorted(
            counts,
            key=lambda path: (counts[path], score_sums.get(path, 0.0)),
            reverse=True,
        )
        dominant_doc = ranked_docs[0]
        dominant_count = counts[dominant_doc]
        second_count = counts[ranked_docs[1]] if len(ranked_docs) > 1 else 0
        dominant_ratio = dominant_count / len(retrieved)

        return (
            doc_path == dominant_doc
            and dominant_count >= 3
            and dominant_ratio >= 0.4
            and dominant_count - second_count >= 1
        )

    def _apply_stop_guard(
        self,
        *,
        question: str,
        retrieved: List[RetrievedChunk],
        decision: MultiActionDecision,
        doc_covered_positions: Dict[str, Set[int]],
    ) -> tuple[MultiActionDecision, bool]:
        if decision.action != ACTION_EXPAND_DOCUMENT:
            return decision, False

        anchor = self._resolve_anchor_chunk(retrieved, decision.anchor_chunk_index)
        if anchor is None:
            return decision, False

        simulated_next = self.retriever.expand_document(
            retrieved,
            k=len(retrieved),
            anchor=anchor,
            covered_positions=doc_covered_positions.get(anchor.doc_path, set()),
        )
        if self._chunk_signature(simulated_next) == self._chunk_signature(retrieved):
            guarded = MultiActionDecision(
                action=ACTION_STOP,
                query=decision.query,
                document_index=decision.document_index,
                anchor_chunk_index=decision.anchor_chunk_index,
                rationale=(
                    f"{decision.rationale} "
                    f"Further document expansion was stopped because it would not change the retrieved chunk set."
                ).strip(),
            )
            return guarded, True

        frontier_available = self.retriever.has_unseen_document_frontier(
            retrieved,
            anchor=anchor,
            covered_positions=doc_covered_positions.get(anchor.doc_path, set()),
        )
        source_score = self._source_evidence(question, anchor.doc_path).score

        if frontier_available or source_score <= 0.0:
            return decision, False

        guarded = MultiActionDecision(
            action=ACTION_STOP,
            query=decision.query,
            document_index=decision.document_index,
            anchor_chunk_index=decision.anchor_chunk_index,
            rationale=(
                f"{decision.rationale} "
                f"Further document expansion was stopped because the explored document frontier appears exhausted."
            ).strip(),
        )
        return guarded, True

    def _search_chunks(self, query: str, top_k: int, doc_path: Optional[str] = None) -> tuple[str, List[RetrievedChunk], float]:
        retrieval_started = perf_counter()
        retrieval_query, _ = self._prepare_retrieval_query(query)
        if doc_path:
            retrieved = self.retriever.retrieve_in_document(retrieval_query, doc_path=doc_path, k=top_k)
        elif self._salient_query_tokens(query):
            retrieved = self.retriever.retrieve(retrieval_query, k=top_k)
        else:
            # Generic clause statements often lack source cues, so prefer broader
            # document coverage before committing to document-local exploration.
            retrieved = self.retriever.retrieve_diverse(retrieval_query, k=top_k, max_per_doc=2)
        retrieval_latency_s = perf_counter() - retrieval_started
        return retrieval_query, retrieved, retrieval_latency_s

    def _run_impl(
        self,
        question: str,
        answer_system_prompt: str,
        top_k: int = RETRIEVAL_TOP_K,
        doc_path: Optional[str] = None,
    ) -> MultiActionAgenticResult:
        current_query = question
        focused_document: Optional[str] = doc_path
        action_history: List[str] = []
        refined_queries: List[str] = []
        iteration_trace: List[dict] = []
        retrieval_latency_s = 0.0
        generation_latency_s = 0.0
        doc_covered_positions: Dict[str, Set[int]] = {}
        controller_steps = 0

        available_actions = CONTRACTNLI_DOC_KNOWN_ACTIONS if doc_path else VALID_ACTIONS

        retrieval_query, retrieved, search_latency_s = self._search_chunks(current_query, top_k, doc_path=doc_path)
        retrieval_latency_s += search_latency_s
        retrieval_steps = 1
        context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        previous_signature = self._chunk_signature(retrieved)
        self._update_doc_coverage(doc_covered_positions, retrieved)
        action_history.append(f"{ACTION_SEARCH_CHUNKS} | query={current_query} | anchor=NONE")

        best_retrieved = list(retrieved)
        best_used = list(used)
        best_context = context
        stop_confirmed = False

        max_controller_steps = max(0, AGENTIC_MAX_ITERS - 1)
        for iteration in range(max_controller_steps):
            controller_steps += 1
            controller_query = current_query
            controller_retrieval_query = retrieval_query
            controller_retrieved = list(retrieved)
            controller_used = list(used)
            controller_context = context
            controller_focused_document = focused_document
            decision, controller_latency_s, raw_controller_output = self._controller_step(
                question=question,
                current_query=current_query,
                focused_document=focused_document,
                available_actions=available_actions,
                action_history=action_history,
                retrieved=retrieved,
            )
            decision, source_guard_applied, anchor_source_score, best_source_score = self._apply_source_guard(
                question=question,
                retrieved=retrieved,
                decision=decision,
                locked_document_path=doc_path,
            )
            decision, stop_guard_applied = self._apply_stop_guard(
                question=question,
                retrieved=retrieved,
                decision=decision,
                doc_covered_positions=doc_covered_positions,
            )
            generation_latency_s += controller_latency_s

            stop_reason: Optional[str] = None
            next_query: Optional[str] = None
            result_query = controller_query
            result_retrieval_query = controller_retrieval_query
            result_focused_document = controller_focused_document
            result_retrieved = controller_retrieved
            result_used = controller_used
            result_context = controller_context
            changed_retrieval = False

            if decision.action == ACTION_STOP:
                stop_confirmed = True
                best_retrieved = list(retrieved)
                best_used = list(used)
                best_context = context
                stop_reason = "controller_stop"
            else:
                if decision.action == ACTION_REFORMULATE_QUERY:
                    candidate_query = self._normalize_optional_text(decision.query) or current_query
                    next_query = candidate_query
                    if self._normalize_query(candidate_query) != self._normalize_query(current_query):
                        refined_queries.append(candidate_query)
                        current_query = candidate_query
                    result_query = current_query
                    result_retrieval_query, next_retrieved, search_latency_s = self._search_chunks(current_query, top_k, doc_path=doc_path)
                    result_focused_document = doc_path
                    retrieval_latency_s += search_latency_s
                    retrieval_steps += 1
                elif decision.action == ACTION_SELECT_DOCUMENT:
                    selected_doc = self._resolve_document_path(retrieved, decision.document_index)
                    if selected_doc is None:
                        next_retrieved = list(retrieved)
                    else:
                        same_doc_chunks = [chunk for chunk in retrieved if chunk.doc_path == selected_doc]
                        anchor = same_doc_chunks[0] if same_doc_chunks else None
                        next_retrieved = self.retriever.focus_document(
                            retrieved,
                            doc_path=selected_doc,
                            k=top_k,
                            anchor=anchor,
                        )
                        result_focused_document = selected_doc
                    retrieval_steps += 1
                elif decision.action == ACTION_EXPAND_DOCUMENT:
                    anchor = self._resolve_anchor_chunk(retrieved, decision.anchor_chunk_index)
                    anchor_doc_path = anchor.doc_path if anchor is not None else ""
                    retrieval_started = perf_counter()
                    next_retrieved = self.retriever.expand_document(
                        retrieved,
                        k=top_k,
                        anchor=anchor,
                        covered_positions=doc_covered_positions.get(anchor_doc_path, set()),
                    )
                    retrieval_latency_s += perf_counter() - retrieval_started
                    retrieval_steps += 1
                    result_query = current_query
                    result_retrieval_query = controller_retrieval_query
                    result_focused_document = anchor_doc_path or focused_document
                else:
                    next_query = self._normalize_optional_text(decision.query) or current_query
                    current_query = next_query
                    result_query = current_query
                    result_retrieval_query, next_retrieved, search_latency_s = self._search_chunks(current_query, top_k, doc_path=doc_path)
                    result_focused_document = doc_path
                    retrieval_latency_s += search_latency_s
                    retrieval_steps += 1

                next_context, next_used = build_context(next_retrieved, char_budget=CONTEXT_CHAR_BUDGET)
                next_signature = self._chunk_signature(next_retrieved)
                result_retrieved = list(next_retrieved)
                result_used = list(next_used)
                result_context = next_context

                if next_signature == previous_signature:
                    stop_reason = "repeated_retrieval"
                else:
                    retrieved = next_retrieved
                    used = next_used
                    context = next_context
                    retrieval_query = result_retrieval_query
                    focused_document = result_focused_document
                    previous_signature = next_signature
                    self._update_doc_coverage(doc_covered_positions, next_retrieved)
                    changed_retrieval = True
                    action_history.append(
                        f"{decision.action} | query={current_query} | document={self._focused_document_text(result_focused_document)} | anchor={decision.anchor_chunk_index or 'NONE'}"
                    )
                    should_stop_early, early_stop_reason = self._should_early_stop_after_retrieval(
                        question=question,
                        retrieved=retrieved,
                        used=used,
                        doc_covered_positions=doc_covered_positions,
                        locked_document_path=doc_path,
                    )
                    if should_stop_early:
                        stop_confirmed = True
                        best_retrieved = list(retrieved)
                        best_used = list(used)
                        best_context = context
                        stop_reason = early_stop_reason

            if self.trace_iterations:
                iteration_trace.append(
                    self._make_iteration_entry(
                        iteration_index=iteration + 1,
                        action=decision.action,
                        controller_query=controller_query,
                        controller_retrieval_query=controller_retrieval_query,
                        controller_retrieved=controller_retrieved,
                        controller_used=controller_used,
                        controller_context=controller_context,
                        stop_reason=stop_reason,
                        next_query=next_query,
                        document_index=decision.document_index,
                        anchor_chunk_index=decision.anchor_chunk_index,
                        controller_latency_s=controller_latency_s,
                        raw_controller_output=raw_controller_output,
                        rationale=decision.rationale,
                        result_query=result_query,
                        result_retrieval_query=result_retrieval_query,
                        focused_document=controller_focused_document,
                        result_focused_document=result_focused_document,
                        result_retrieved=result_retrieved,
                        result_used=result_used,
                        result_context=result_context,
                        changed_retrieval=changed_retrieval,
                        source_guard_applied=source_guard_applied,
                        anchor_source_score=anchor_source_score,
                        best_source_score=best_source_score,
                        stop_guard_applied=stop_guard_applied,
                    )
                )

            if stop_reason is not None:
                break

        final_context = best_context if stop_confirmed else context
        final_retrieved = best_retrieved if stop_confirmed else retrieved
        final_used = best_used if stop_confirmed else used

        user_prompt = build_user_prompt(context=final_context, question=question)
        generation_started = perf_counter()
        final_system_prompt = answer_system_prompt
        if not stop_confirmed:
            final_system_prompt = (
                answer_system_prompt
                + " The retrieval controller did not explicitly confirm that the evidence is sufficient. "
                + "If the context does not directly support the answer, say you don't know."
            )
        answer = self.model.generate(final_system_prompt, user_prompt)
        generation_latency_s += perf_counter() - generation_started

        return MultiActionAgenticResult(
            answer=answer,
            retrieved=final_retrieved,
            used=final_used,
            iterations=controller_steps,
            retrieval_steps=retrieval_steps,
            action_history=action_history,
            refined_queries=refined_queries,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(final_context),
            prompt_tokens=self.model.count_tokens(final_system_prompt) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
            iteration_trace=iteration_trace,
        )

    def _update_doc_coverage(self, coverage: Dict[str, Set[int]], chunks: List[RetrievedChunk]) -> None:
        docs = {chunk.doc_path for chunk in chunks}
        for doc_path in docs:
            positions = self.retriever.positions_for_doc_chunks(chunks, doc_path)
            if not positions:
                continue
            coverage.setdefault(doc_path, set()).update(positions)

    def _should_early_stop_after_retrieval(
        self,
        *,
        question: str,
        retrieved: List[RetrievedChunk],
        used: List[RetrievedChunk],
        doc_covered_positions: Dict[str, Set[int]],
        locked_document_path: Optional[str] = None,
    ) -> tuple[bool, str]:
        if not retrieved:
            return False, ""

        dominant_doc, dominant_ratio = self._dominant_doc_stats(retrieved)
        if dominant_doc is None or dominant_ratio < 0.8:
            return False, ""

        anchor = retrieved[0]
        source_score = 1.0 if locked_document_path else self._source_evidence(question, anchor.doc_path).score
        if source_score < 0.5:
            return False, ""

        simulated_next = self.retriever.expand_document(
            retrieved,
            k=len(retrieved),
            anchor=anchor,
            covered_positions=doc_covered_positions.get(anchor.doc_path, set()),
        )
        current_used_signature = self._chunk_signature(used)
        simulated_context, simulated_used = build_context(simulated_next, char_budget=CONTEXT_CHAR_BUDGET)
        simulated_used_signature = self._chunk_signature(simulated_used)

        if simulated_used_signature == current_used_signature:
            return True, "marginal_utility_exhausted"

        if not self.retriever.has_unseen_document_frontier(
            retrieved,
            anchor=anchor,
            covered_positions=doc_covered_positions.get(anchor.doc_path, set()),
        ):
            return True, "document_frontier_exhausted"

        if not simulated_context.strip():
            return True, "empty_future_context"

        return False, ""

    def run(self, question: str, top_k: int = RETRIEVAL_TOP_K, doc_path: Optional[str] = None) -> MultiActionAgenticResult:
        return self._run_impl(question=question, answer_system_prompt=SYSTEM_PROMPT, top_k=top_k, doc_path=doc_path)

    def run_label(self, question: str, top_k: int = RETRIEVAL_TOP_K, doc_path: Optional[str] = None) -> MultiActionAgenticResult:
        return self._run_impl(question=question, answer_system_prompt=LABEL_SYSTEM_PROMPT, top_k=top_k, doc_path=doc_path)

    def run_hotpot(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> MultiActionAgenticResult:
        return self._run_impl(question=question, answer_system_prompt=HOTPOT_SYSTEM_PROMPT, top_k=top_k)
