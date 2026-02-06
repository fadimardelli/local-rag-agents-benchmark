from dataclasses import dataclass
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
from src.rag.prompts import SYSTEM_PROMPT, USER_TEMPLATE
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


class AgenticRAG:
    def __init__(self):
        self.retriever = Retriever()
        self.model = LlamaCppModel(
            model_path=LLAMA_GGUF_PATH,
            n_ctx=LLAMA_N_CTX,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
        )

    def _assess_sufficiency(self, question: str, context: str) -> bool:
        user_prompt = USER_TEMPLATE.format(context=context, question=question)
        result = self.model.generate(ASSESS_PROMPT, user_prompt)
        return result.strip().upper().startswith("SUFFICIENT")

    def _refine_query(self, question: str, context: str) -> str:
        user_prompt = USER_TEMPLATE.format(context=context, question=question)
        refined = self.model.generate(REFINE_PROMPT, user_prompt)
        return refined.strip()

    def run(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> AgenticResult:
        refined_queries: List[str] = []
        retrieved: List[RetrievedChunk] = []
        used: List[RetrievedChunk] = []
        current_query = question

        for i in range(AGENTIC_MAX_ITERS):
            retrieved = self.retriever.retrieve(current_query, k=top_k)
            context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)

            if self._assess_sufficiency(question, context):
                break

            if i < AGENTIC_MAX_ITERS - 1:
                refined = self._refine_query(question, context)
                refined_queries.append(refined)
                current_query = refined

        user_prompt = USER_TEMPLATE.format(context=context, question=question)
        answer = self.model.generate(SYSTEM_PROMPT, user_prompt)

        return AgenticResult(
            answer=answer,
            retrieved=retrieved,
            used=used,
            iterations=len(refined_queries) + 1,
            refined_queries=refined_queries,
        )
