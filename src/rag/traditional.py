from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from src.config.defaults import (
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


@dataclass(frozen=True)
class RAGResult:
    answer: str
    retrieved: List[RetrievedChunk]
    used: List[RetrievedChunk]
    retrieval_latency_s: float
    generation_latency_s: float
    retrieved_context_tokens: int
    prompt_tokens: int
    answer_tokens: int


class TraditionalRAG:
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

    def run(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResult:
        retrieval_started = perf_counter()
        retrieved = self.retriever.retrieve(question, k=top_k)
        retrieval_latency_s = perf_counter() - retrieval_started
        context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        user_prompt = build_user_prompt(context=context, question=question)
        generation_started = perf_counter()
        answer = self.model.generate(SYSTEM_PROMPT, user_prompt)
        generation_latency_s = perf_counter() - generation_started
        return RAGResult(
            answer=answer,
            retrieved=retrieved,
            used=used,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(context),
            prompt_tokens=self.model.count_tokens(SYSTEM_PROMPT) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
        )

    def run_label(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResult:
        retrieval_started = perf_counter()
        retrieved = self.retriever.retrieve(question, k=top_k)
        retrieval_latency_s = perf_counter() - retrieval_started
        context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        user_prompt = build_user_prompt(context=context, question=question)
        generation_started = perf_counter()
        answer = self.model.generate(LABEL_SYSTEM_PROMPT, user_prompt)
        generation_latency_s = perf_counter() - generation_started
        return RAGResult(
            answer=answer,
            retrieved=retrieved,
            used=used,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(context),
            prompt_tokens=self.model.count_tokens(LABEL_SYSTEM_PROMPT) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
        )

    def run_hotpot(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResult:
        retrieval_started = perf_counter()
        retrieved = self.retriever.retrieve(question, k=top_k)
        retrieval_latency_s = perf_counter() - retrieval_started
        context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        user_prompt = build_user_prompt(context=context, question=question)
        generation_started = perf_counter()
        answer = self.model.generate(HOTPOT_SYSTEM_PROMPT, user_prompt)
        generation_latency_s = perf_counter() - generation_started
        return RAGResult(
            answer=answer,
            retrieved=retrieved,
            used=used,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieved_context_tokens=self.model.count_tokens(context),
            prompt_tokens=self.model.count_tokens(HOTPOT_SYSTEM_PROMPT) + self.model.count_tokens(user_prompt),
            answer_tokens=self.model.count_tokens(answer),
        )
