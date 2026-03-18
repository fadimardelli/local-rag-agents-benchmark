from dataclasses import dataclass
from pathlib import Path
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
        retrieved = self.retriever.retrieve(question, k=top_k)
        context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        user_prompt = build_user_prompt(context=context, question=question)
        answer = self.model.generate(SYSTEM_PROMPT, user_prompt)
        return RAGResult(answer=answer, retrieved=retrieved, used=used)

    def run_label(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResult:
        retrieved = self.retriever.retrieve(question, k=top_k)
        context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        user_prompt = build_user_prompt(context=context, question=question)
        answer = self.model.generate(LABEL_SYSTEM_PROMPT, user_prompt)
        return RAGResult(answer=answer, retrieved=retrieved, used=used)

    def run_hotpot(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResult:
        retrieved = self.retriever.retrieve(question, k=top_k)
        context, used = build_context(retrieved, char_budget=CONTEXT_CHAR_BUDGET)
        user_prompt = build_user_prompt(context=context, question=question)
        answer = self.model.generate(HOTPOT_SYSTEM_PROMPT, user_prompt)
        return RAGResult(answer=answer, retrieved=retrieved, used=used)
