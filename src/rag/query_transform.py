from dataclasses import dataclass

from src.config.defaults import QUERY_TRANSFORM_MAX_TOKENS, QUERY_TRANSFORM_MODE
from src.rag.llama_cpp import LlamaCppModel
from src.rag.prompts import QUERY_TRANSFORM_SYSTEM_PROMPT, build_query_transform_prompt


@dataclass(frozen=True)
class QueryTransformResult:
    retrieval_query: str
    latency_s: float
    raw_output: str


def normalize_transform_output(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def build_retrieval_query(
    *,
    question: str,
    model: LlamaCppModel,
    mode: str = QUERY_TRANSFORM_MODE,
    max_tokens: int = QUERY_TRANSFORM_MAX_TOKENS,
) -> QueryTransformResult:
    if mode == "none":
        return QueryTransformResult(retrieval_query=question, latency_s=0.0, raw_output=question)
    if mode != "hyde":
        raise ValueError(f"Unknown query transform mode '{mode}'")

    from time import perf_counter

    user_prompt = build_query_transform_prompt(question)
    started = perf_counter()
    raw_output = model.generate(
        QUERY_TRANSFORM_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    latency_s = perf_counter() - started
    retrieval_query = normalize_transform_output(raw_output) or question
    return QueryTransformResult(
        retrieval_query=retrieval_query,
        latency_s=latency_s,
        raw_output=raw_output,
    )
