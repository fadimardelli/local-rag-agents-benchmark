from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from src.config.defaults import (
    AGENTIC_MAX_ITERS,
    CONTEXT_CHAR_BUDGET,
    LLAMA_GGUF_PATH,
    LLAMA_MAX_TOKENS,
    LLAMA_N_CTX,
    LLAMA_TEMPERATURE,
    RETRIEVAL_TOP_K,
)
from src.rag.context_builder import build_context
from src.rag.llama_cpp import LlamaCppModel
from src.rag.prompts import HOTPOT_SYSTEM_PROMPT, LABEL_SYSTEM_PROMPT, SYSTEM_PROMPT, build_user_prompt
from src.rag.retriever import RetrievedChunk, Retriever


ASSESS_PROMPT = (
    "You are assessing whether the provided context is sufficient to answer the question. "
    "Reply with exactly one word: SUFFICIENT or INSUFFICIENT."
)

REFINE_PROMPT = (
    "Rewrite the question to improve document retrieval. "
    "Return a short, focused search query. "
    "Do not answer the question."
)


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


class AgenticRAG:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        model: Optional[LlamaCppModel] = None,
        model_path: Optional[Path] = None,
    ):
        self.retriever = retriever or Retriever()
        self.model = model or LlamaCppModel(
            model_path=model_path or LLAMA_GGUF_PATH,
            n_ctx=LLAMA_N_CTX,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
        )

    def _assess_sufficiency(self, question: str, context: str) -> bool:
        user_prompt = build_user_prompt(context=context, question=question)
        result = self.model.generate(ASSESS_PROMPT, user_prompt)
        return result.strip().upper().startswith("SUFFICIENT")

    def _refine_query(self, question: str, context: str) -> str:
        user_prompt = build_user_prompt(context=context, question=question)
        refined = self.model.generate(REFINE_PROMPT, user_prompt)
        return refined.strip()

    def run(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        refined_queries: List[str] = []
        retrieved: List[RetrievedChunk] = []
        used: List[RetrievedChunk] = []
        current_query = question
        retrieval_latency_s = 0.0

        for i in range(AGENTIC_MAX_ITERS):
            retrieval_started = perf_counter()
            retrieved = self.retriever.retrieve(current_query, k=top_k)
            retrieval_latency_s += perf_counter() - retrieval_started
            context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)

            if self._assess_sufficiency(question, context):
                break

            if i < AGENTIC_MAX_ITERS - 1:
                refined = self._refine_query(question, context)
                refined_queries.append(refined)
                current_query = refined

        user_prompt = build_user_prompt(context=context, question=question)
        generation_started = perf_counter()
        answer = self.model.generate(SYSTEM_PROMPT, user_prompt)
        generation_latency_s = perf_counter() - generation_started

        return AgenticResult(
            answer=answer,
            retrieved=retrieved,
            used=used,
            iterations=len(refined_queries) + 1,
            refined_queries=refined_queries,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(context),
            prompt_tokens=self.model.count_tokens(SYSTEM_PROMPT) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
        )

    def run_label(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        refined_queries: List[str] = []
        retrieved: List[RetrievedChunk] = []
        used: List[RetrievedChunk] = []
        current_query = question
        retrieval_latency_s = 0.0

        for i in range(AGENTIC_MAX_ITERS):
            retrieval_started = perf_counter()
            retrieved = self.retriever.retrieve(current_query, k=top_k)
            retrieval_latency_s += perf_counter() - retrieval_started
            context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)

            if self._assess_sufficiency(question, context):
                break

            if i < AGENTIC_MAX_ITERS - 1:
                refined = self._refine_query(question, context)
                refined_queries.append(refined)
                current_query = refined

        user_prompt = build_user_prompt(context=context, question=question)
        generation_started = perf_counter()
        answer = self.model.generate(LABEL_SYSTEM_PROMPT, user_prompt)
        generation_latency_s = perf_counter() - generation_started

        return AgenticResult(
            answer=answer,
            retrieved=retrieved,
            used=used,
            iterations=len(refined_queries) + 1,
            refined_queries=refined_queries,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(context),
            prompt_tokens=self.model.count_tokens(LABEL_SYSTEM_PROMPT) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
        )

    def run_hotpot(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        refined_queries: List[str] = []
        retrieved: List[RetrievedChunk] = []
        used: List[RetrievedChunk] = []
        current_query = question
        retrieval_latency_s = 0.0

        for i in range(AGENTIC_MAX_ITERS):
            retrieval_started = perf_counter()
            retrieved = self.retriever.retrieve(current_query, k=top_k)
            retrieval_latency_s += perf_counter() - retrieval_started
            context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)

            if self._assess_sufficiency(question, context):
                break

            if i < AGENTIC_MAX_ITERS - 1:
                refined = self._refine_query(question, context)
                refined_queries.append(refined)
                current_query = refined

        user_prompt = build_user_prompt(context=context, question=question)
        generation_started = perf_counter()
        answer = self.model.generate(HOTPOT_SYSTEM_PROMPT, user_prompt)
        generation_latency_s = perf_counter() - generation_started

        return AgenticResult(
            answer=answer,
            retrieved=retrieved,
            used=used,
            iterations=len(refined_queries) + 1,
            refined_queries=refined_queries,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(context),
            prompt_tokens=self.model.count_tokens(HOTPOT_SYSTEM_PROMPT) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
        )
