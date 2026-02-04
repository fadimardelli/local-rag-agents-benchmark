from dataclasses import dataclass
from typing import List

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
from src.rag.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.rag.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RAGResult:
    answer: str
    retrieved: List[RetrievedChunk]
    used: List[RetrievedChunk]


class TraditionalRAG:
    def __init__(self):
        self.retriever = Retriever()
        self.model = LlamaCppModel(
            model_path=LLAMA_GGUF_PATH,
            n_ctx=LLAMA_N_CTX,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
        )

    def run(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResult:
        retrieved = self.retriever.retrieve(question, k=top_k)
        context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        user_prompt = USER_TEMPLATE.format(context=context, question=question)
        answer = self.model.generate(SYSTEM_PROMPT, user_prompt)
        return RAGResult(answer=answer, retrieved=retrieved, used=used)
