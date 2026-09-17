from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Iterable

import streamlit as st

from src.config.defaults import (
    AGENTIC_CORRECTIVE_TOP_K,
    AGENTIC_MAX_ITERS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXT_CHAR_BUDGET,
    CONTRACTNLI_INDEX_PATH,
    CONTRACTNLI_META_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
    HOTPOTQA_INDEX_PATH,
    HOTPOTQA_META_PATH,
    LLAMA_GGUF_PATH_3B,
    LLAMA_GGUF_PATH_8B,
    LLAMA_MAX_TOKENS,
    LLAMA_N_CTX,
    LLAMA_N_GPU_LAYERS,
    LLAMA_SEED,
    LLAMA_TEMPERATURE,
)
from src.data.chunking import chunk_text
from src.indexing import EmbeddingModel, build_faiss_index, save_faiss_index, save_metadata
from src.rag.agentic import AgenticRAG, AgenticResult
from src.rag.llama_cpp import LlamaCppModel
from src.rag.retriever import RetrievedChunk, Retriever
from src.rag.traditional import RAGResult, TraditionalRAG


APP_TITLE = "Local RAG Assistant"
USER_INDEX_ROOT = Path("data/user_indexes")
BUILTIN_CORPORA = {
    "LegalBench-mini corpus": {
        "index_path": CONTRACTNLI_INDEX_PATH,
        "meta_path": CONTRACTNLI_META_PATH,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "kind": "built-in",
    },
    "HotpotQA corpus": {
        "index_path": HOTPOTQA_INDEX_PATH,
        "meta_path": HOTPOTQA_META_PATH,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "kind": "built-in",
    },
}

MODEL_ONLY_SYSTEM_PROMPT = (
    "You are a concise local assistant. Answer from your own model knowledge only. "
    "If you are unsure, say so. Do not claim that you checked local documents."
)

MODEL_OPTIONS = {
    "Llama 3.2 3B": LLAMA_GGUF_PATH_3B,
    "Llama 3.1 8B": LLAMA_GGUF_PATH_8B,
    "Qwen 2.5 3B": Path("/Users/fadimardelli/models/qwen2.5-3b-instruct-q4_k_m.gguf"),
    "Qwen 2.5 7B": Path("/Users/fadimardelli/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
}

EMBEDDING_MODEL_OPTIONS = {
    "BGE base English v1.5": "BAAI/bge-base-en-v1.5",
    "BGE small English v1.5": "BAAI/bge-small-en-v1.5",
    "BGE M3": "BAAI/bge-m3",
}

EXAMPLE_QUESTIONS = {
    "LegalBench-mini corpus": {
        "Slow corrective: COVID-19 IP licenses": (
            "Consider the Intellectual Property Agreement for COVID-19 Treatment Development between "
            "Marv Enterprises, LLC, Premier Biomedical, Inc., and Technology Health, Inc.; "
            "What licenses are granted under this contract?"
        ),
        "Slow corrective: Viber data sharing": (
            'Consider "Viber Messenger"\'s privacy policy; how is any information collected by viber shared with other parties?'
        ),
        "Slow corrective: Material adverse effect": (
            'Consider the Acquisition Agreement between Parent "REALTY INCOME CORPORATION" and Target "VEREIT, INC."; '
            'What is the Definition of "Material Adverse Effect"'
        ),
        "Fiverr privacy policy: task visibility": (
            'Consider "Fiverr"\'s privacy policy; who can see which tasks i hire workers for?'
        ),
        "Fiverr privacy policy: hacker protection": (
            'Consider "Fiverr"\'s privacy policy; how is my info protected from hackers?'
        ),
        "Keep privacy policy: physical measurements": (
            'Consider "Keep"\'s privacy policy; do you keep track of my physical measurements like height and weight?'
        ),
        "Motorola NDA: rights to confidential information": (
            "Consider Motorola's Non-Disclosure Agreement; Does the document indicate that the Agreement does not grant "
            "the Receiving Party any rights to the Confidential Information?"
        ),
    },
    "HotpotQA corpus": {
        "Slow corrective: Pat Hingle / Western TV": (
            'Martin Patterson "Pat" Hingle was a close friend of an actor who achieved success in what Western TV series?'
        ),
        "Slow corrective: War Chhod Na Yaar actor": (
            "Which lead actor/actress in War Chhod Na Yaar has also acted in Bengali and English-language films?"
        ),
        "Slow corrective: Black Maverick biography": (
            '"Black Maverick" is a biography of what American civil rights leader, fraternal organization leader, '
            "entrepreneur and surgeon?"
        ),
        "Laleli Mosque / Esma Sultan Mansion": (
            "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"
        ),
        "Science fantasy YA series": (
            "What science fantasy young adult series, told in first person, has a set of companion books narrating "
            "the stories of enslaved worlds and alien species?"
        ),
        "Corliss Archer / Kiss and Tell": (
            "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
        ),
        "Scott Derrickson / Ed Wood": (
            "Were Scott Derrickson and Ed Wood of the same nationality?"
        ),
    },
}


@dataclass(frozen=True)
class DemoRun:
    label: str
    answer: str
    retrieved: list[RetrievedChunk]
    used: list[RetrievedChunk]
    latency_s: float
    retrieval_calls: int
    iterations: int
    retrieval_latency_s: float
    generation_latency_s: float
    retrieved_context_tokens: int
    prompt_tokens: int
    answer_tokens: int
    routing_type: str | None = None
    refined_queries: tuple[str, ...] = ()
    iteration_trace: tuple[dict, ...] = ()
    answer_selection_source: str | None = None
    first_pass_answer: str | None = None
    corrected_pass_answer: str | None = None


@dataclass(frozen=True)
class UploadedDocument:
    source_name: str
    storage_name: str
    text: str


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #102033;
            --muted: #5b6778;
            --line: #d8e0ea;
            --accent: #1f5f82;
            --accent-soft: #eaf3f7;
            --surface: #ffffff;
            --bg: #f6f3ea;
        }
        .stApp {
            background: var(--bg);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3, p, label, span {
            color: var(--ink);
        }
        header,
        #MainMenu,
        footer,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stDeployButton"],
        .stDeployButton {
            display: none !important;
        }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.55rem 0.7rem;
            box-shadow: none;
        }
        div[data-testid="stMetric"] * {
            color: var(--ink) !important;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted) !important;
        }
        div[data-testid="stExpander"] {
            background: var(--surface);
            border-color: var(--line);
            border-radius: 8px;
        }
        .stButton > button,
        div[data-testid="stButton"] button {
            color: var(--ink) !important;
            background: var(--surface) !important;
            border: 1px solid var(--line) !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }
        .stButton > button[kind="primary"],
        div[data-testid="stButton"] button[kind="primary"] {
            color: #ffffff !important;
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        .stButton > button *,
        div[data-testid="stButton"] button * {
            color: inherit !important;
        }
        .answer-card {
            border: 1px solid var(--line);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.25rem 0 0.55rem 0;
            background: var(--surface);
            color: var(--ink) !important;
            line-height: 1.55;
        }
        .answer-card * {
            color: var(--ink) !important;
        }
        .answer-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 800;
            color: var(--accent) !important;
            margin-bottom: 0.35rem;
        }
        .status-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            margin: 0.55rem 0 0.8rem 0;
            background: var(--surface);
            color: var(--ink) !important;
        }
        .status-title {
            font-weight: 800;
            margin-bottom: 0.2rem;
            color: var(--ink) !important;
        }
        .status-body {
            color: var(--muted) !important;
            font-size: 0.92rem;
        }
        .section-kicker {
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.74rem;
            color: var(--accent);
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.15rem 0 0.75rem 0;
        }
        .info-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: var(--surface);
            color: var(--ink) !important;
            padding: 0.22rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 750;
            white-space: nowrap;
        }
        .info-badge strong {
            color: var(--accent) !important;
            font-weight: 850;
        }
        .info-badge.accent {
            background: var(--accent-soft);
            border-color: #b6d6dc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._-").lower()
    return slug or "custom_corpus"


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(name).name).strip("._-") or "document.txt"


def _text_storage_filename(source_name: str) -> str:
    if Path(source_name).suffix.lower() == ".pdf":
        return _safe_filename(f"{source_name}.txt")
    return source_name


def _hf_cache_dir_for_model(model_name: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"


def _is_embedding_model_available(model_name: str) -> bool:
    cache_dir = _hf_cache_dir_for_model(model_name)
    snapshots_dir = cache_dir / "snapshots"
    return snapshots_dir.exists() and any(snapshots_dir.iterdir())


def _available_embedding_model_options() -> dict[str, str]:
    available = {
        label: model_name
        for label, model_name in EMBEDDING_MODEL_OPTIONS.items()
        if _is_embedding_model_available(model_name)
    }
    if available:
        return available
    default_label = next(
        (
            label
            for label, model_name in EMBEDDING_MODEL_OPTIONS.items()
            if model_name == EMBEDDING_MODEL_NAME
        ),
        EMBEDDING_MODEL_NAME,
    )
    return {default_label: EMBEDDING_MODEL_NAME}


def _strip_rtf(text: str) -> str:
    """Best-effort cleanup for TextEdit-style RTF files saved with a .txt suffix."""
    text = re.sub(r"{\\fonttbl.*?}", " ", text, flags=re.DOTALL)
    text = re.sub(r"{\\colortbl.*?}", " ", text, flags=re.DOTALL)
    text = re.sub(r"{\\[*][^{}]*}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\par(?![a-zA-Z])", "\n", text)
    text = re.sub(r"\\pard(?![a-zA-Z])", "\n", text)
    text = re.sub(r"\\tab(?![a-zA-Z])", " ", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
    text = text.replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")
    text = re.sub(r"\s\\\s", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


def _clean_document_text(text: str) -> str:
    if text.lstrip().startswith("{\\rtf") or "\\rtf" in text[:200]:
        return _strip_rtf(text)
    return text


def _extract_pdf_text(pdf_bytes: bytes, filename: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError(
            f"{filename}: PDF support requires pypdf. Run `pip install -r requirements.txt` and restart Streamlit."
        ) from exc

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError(f"{filename}: encrypted PDFs are not supported.") from exc
        page_texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                raise ValueError(f"{filename}: could not extract text from page {page_number}.") from exc
            if page_text.strip():
                page_texts.append(page_text.strip())
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{filename}: could not read PDF.") from exc

    text = "\n\n".join(page_texts)
    if not text.strip():
        raise ValueError(f"{filename}: no extractable text found. Scanned/image-only PDFs need OCR first.")
    return text


def _list_custom_corpora() -> dict[str, dict[str, object]]:
    corpora: dict[str, dict[str, object]] = {}
    if not USER_INDEX_ROOT.exists():
        return corpora
    for corpus_dir in sorted(USER_INDEX_ROOT.iterdir()):
        index_path = corpus_dir / "index.faiss"
        meta_path = corpus_dir / "metadata.jsonl"
        manifest_path = corpus_dir / "manifest.json"
        if not index_path.exists() or not meta_path.exists():
            continue
        display_name = corpus_dir.name
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                display_name = manifest.get("display_name") or display_name
            except json.JSONDecodeError:
                manifest = {}
        corpora[f"Custom: {display_name}"] = {
            "index_path": index_path,
            "meta_path": meta_path,
            "embedding_model": manifest.get("embedding_model") or EMBEDDING_MODEL_NAME,
            "manifest_path": manifest_path,
            "kind": "custom",
            "manifest": manifest,
        }
    return corpora


def _corpus_registry() -> dict[str, dict[str, object]]:
    return {**BUILTIN_CORPORA, **_list_custom_corpora()}


@st.cache_resource(show_spinner=False)
def load_embedder(embedding_model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingModel:
    return EmbeddingModel(embedding_model_name, normalize=EMBEDDING_NORMALIZE)



@st.cache_resource(show_spinner=False)
def load_model(model_path_text: str) -> LlamaCppModel:
    model_path = Path(model_path_text).expanduser()
    return LlamaCppModel(
        model_path=model_path,
        n_ctx=LLAMA_N_CTX,
        n_gpu_layers=LLAMA_N_GPU_LAYERS,
        seed=LLAMA_SEED,
        temperature=LLAMA_TEMPERATURE,
        max_tokens=LLAMA_MAX_TOKENS,
    )


@st.cache_resource(show_spinner=False)
def load_retriever(index_path_text: str, meta_path_text: str, embedding_model_name: str) -> Retriever:
    return Retriever(
        index_path=Path(index_path_text),
        meta_path=Path(meta_path_text),
        embedding_model_name=embedding_model_name,
    )


def _run_model_only(model: LlamaCppModel, question: str) -> DemoRun:
    model.reset()
    started = perf_counter()
    answer = model.generate(MODEL_ONLY_SYSTEM_PROMPT, question)
    latency_s = perf_counter() - started
    return DemoRun(
        label="Local model",
        answer=answer,
        retrieved=[],
        used=[],
        latency_s=latency_s,
        retrieval_calls=0,
        iterations=0,
        retrieval_latency_s=0.0,
        generation_latency_s=latency_s,
        retrieved_context_tokens=0,
        prompt_tokens=model.count_tokens(MODEL_ONLY_SYSTEM_PROMPT) + model.count_tokens(question),
        answer_tokens=model.count_tokens(answer),
    )


def _to_demo_run(label: str, out: RAGResult | AgenticResult, latency_s: float) -> DemoRun:
    return DemoRun(
        label=label,
        answer=out.answer,
        retrieved=list(out.retrieved),
        used=list(out.used),
        latency_s=latency_s,
        retrieval_calls=getattr(out, "retrieval_calls", 1),
        iterations=getattr(out, "iterations", 1),
        retrieval_latency_s=out.retrieval_latency_s,
        generation_latency_s=out.generation_latency_s,
        retrieved_context_tokens=out.retrieved_context_tokens,
        prompt_tokens=out.prompt_tokens,
        answer_tokens=out.answer_tokens,
        routing_type=getattr(out, "routing_type", None),
        refined_queries=tuple(getattr(out, "refined_queries", []) or []),
        iteration_trace=tuple(getattr(out, "iteration_trace", []) or []),
        answer_selection_source=getattr(out, "answer_selection_source", None),
        first_pass_answer=getattr(out, "first_pass_answer", None),
        corrected_pass_answer=getattr(out, "corrected_pass_answer", None),
    )


def _run_traditional(
    *,
    corpus_label: str,
    model: LlamaCppModel,
    retriever: Retriever,
    question: str,
    top_k: int,
) -> DemoRun:
    rag = TraditionalRAG(retriever=retriever, model=model)
    model.reset()
    started = perf_counter()
    out = rag.run_hotpot(question, top_k=top_k) if corpus_label == "HotpotQA corpus" else rag.run(question, top_k=top_k)
    return _to_demo_run("Traditional RAG", out, perf_counter() - started)


def _run_corrective(
    *,
    corpus_label: str,
    model: LlamaCppModel,
    retriever: Retriever,
    question: str,
    top_k: int,
) -> DemoRun:
    rag = AgenticRAG(retriever=retriever, model=model, trace_iterations=True)
    model.reset()
    started = perf_counter()
    out = rag.run_hotpot(question, top_k=top_k) if corpus_label == "HotpotQA corpus" else rag.run(question, top_k=top_k)
    return _to_demo_run("Bounded Corrective RAG", out, perf_counter() - started)


def _short_text(text: str, max_chars: int = 900) -> str:
    cleaned = " ".join(_clean_document_text(text).strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _chunk_title(chunk: RetrievedChunk, idx: int) -> str:
    return f"{idx}. {Path(chunk.doc_path).name} · chars {chunk.start}-{chunk.end}"


def _context_char_count(chunks: Iterable[RetrievedChunk]) -> int:
    total = 0
    for idx, chunk in enumerate(chunks):
        text = chunk.text.strip()
        if not text:
            continue
        total += len(text) + (2 if idx > 0 else 0)
    return total


def _source_names(chunks: Iterable[RetrievedChunk], *, limit: int = 3) -> str:
    names: list[str] = []
    for chunk in chunks:
        name = Path(chunk.doc_path).name
        if name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    if not names:
        return "-"
    suffix = "" if len(names) < limit else "+"
    return ", ".join(names) + suffix


def _source_summary_rows(chunks: Iterable[RetrievedChunk], *, limit: int = 8) -> list[dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for rank, chunk in enumerate(chunks, start=1):
        source = Path(chunk.doc_path).name
        row = summary.setdefault(
            source,
            {
                "source": source,
                "chunks": 0,
                "first_rank": rank,
            },
        )
        row["chunks"] = int(row["chunks"]) + 1
        row["first_rank"] = min(int(row["first_rank"]), rank)
    rows = sorted(summary.values(), key=lambda row: (int(row["first_rank"]), -int(row["chunks"])))
    return rows[:limit]


def _render_source_summary(chunks: Iterable[RetrievedChunk], *, title: str) -> None:
    rows = _source_summary_rows(chunks)
    if not rows:
        return
    with st.expander(title, expanded=False):
        st.dataframe(rows, hide_index=True, width="stretch")


def _render_answer(answer: str) -> None:
    safe_answer = escape(answer or "No answer returned.").replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-label">Answer</div>
            <div>{safe_answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_run_badges(run: DemoRun) -> None:
    badges = [
        ("Runtime", f"{run.latency_s:.2f}s", "accent"),
        ("Prompt", f"{run.prompt_tokens} tokens", ""),
        ("Answer", f"{run.answer_tokens} tokens", ""),
    ]
    if run.retrieved or run.used:
        correction = "-"
        if run.label == "Bounded Corrective RAG":
            correction = "activated" if run.iterations > 1 else "not needed"
        elif run.label == "Traditional RAG":
            correction = "standard retrieval"
        badges.extend(
            [
                ("Retrieval", f"{run.retrieval_calls} calls", "accent" if run.retrieval_calls > 1 else ""),
                ("Evidence", f"{len(run.used)} chunks", ""),
                ("Correction", correction, ""),
                ("Sources", _source_names(run.used), ""),
            ]
        )
    badge_html = "".join(
        f'<span class="info-badge {css_class}"><strong>{escape(label)}</strong> {escape(value)}</span>'
        for label, value, css_class in badges
    )
    st.markdown(f'<div class="badge-row">{badge_html}</div>', unsafe_allow_html=True)


def _render_run_metrics(run: DemoRun) -> None:
    if not run.retrieved and not run.used:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Live runtime", f"{run.latency_s:.2f}s")
        metric_cols[1].metric("Prompt tokens", str(run.prompt_tokens))
        metric_cols[2].metric("Answer tokens", str(run.answer_tokens))
        return

    metric_cols = st.columns(5)
    metric_cols[0].metric("Live runtime", f"{run.latency_s:.2f}s")
    metric_cols[1].metric("Retrieval calls", str(run.retrieval_calls))
    metric_cols[2].metric("Iterations", str(run.iterations))
    metric_cols[3].metric("Prompt tokens", str(run.prompt_tokens))
    metric_cols[4].metric("Answer tokens", str(run.answer_tokens))

    context_cols = st.columns(3)
    context_cols[0].metric("Candidate chunks", str(len(run.retrieved)))
    context_cols[1].metric("Evidence chunks", str(len(run.used)))
    context_cols[2].metric("Evidence chars", f"{_context_char_count(run.used):,}")


def _render_status(run: DemoRun) -> None:
    if run.label == "Traditional RAG":
        title = "One-pass retrieval"
        body = "The final prompt context is built directly from the initial retrieval results."
    elif run.iterations > 1:
        backup = "with backup retrieval" if run.retrieval_calls >= 3 else "without backup retrieval"
        title = "Correction activated"
        body = f"The run used {run.retrieval_calls} retrieval calls and fused the resulting ranked candidates {backup}."
    else:
        title = "Correction not activated"
        body = "The controller stopped after the first retrieved context."
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">{escape(title)}</div>
            <div class="status-body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plain_correction_steps(run: DemoRun) -> list[str]:
    if run.label != "Bounded Corrective RAG":
        return []
    if run.iterations <= 1:
        return ["The controller accepted the first retrieved context, so no corrective retrieval was run."]

    steps = ["The controller triggered one corrective stage after the first retrieval pass."]
    if run.refined_queries:
        steps.append(f"Rewrite used: {run.refined_queries[0]}")
    if len(run.refined_queries) > 1 or run.retrieval_calls >= 3:
        steps.append("A backup retrieval was also used.")
    elif run.retrieval_calls >= 2:
        steps.append("The corrective stage used one additional retrieval call.")
    if run.routing_type:
        route = run.routing_type.replace("_", " ")
        steps.append(f"Retrieval route: {route}.")
    if run.answer_selection_source:
        steps.append(f"Final answer selected from: {run.answer_selection_source}.")
    return steps


def _render_plain_correction_summary(run: DemoRun) -> None:
    steps = _plain_correction_steps(run)
    if not steps:
        return
    with st.expander("Correction summary", expanded=False):
        for step in steps:
            st.markdown(f"- {step}")


def _read_uploaded_document(uploaded_file) -> UploadedDocument:
    filename = _safe_filename(uploaded_file.name)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise ValueError(f"{filename}: only .txt, .md, and .pdf files are supported.")
    contents = uploaded_file.getvalue()
    if suffix == ".pdf":
        text = _extract_pdf_text(contents, filename)
    else:
        text = contents.decode("utf-8", errors="replace")
        text = _clean_document_text(text)
    if not text.strip():
        raise ValueError(f"{filename}: file is empty.")
    return UploadedDocument(
        source_name=filename,
        storage_name=_text_storage_filename(filename),
        text=text,
    )


def _build_custom_corpus(
    *,
    display_name: str,
    uploaded_files,
    chunk_size: int,
    overlap: int,
    embedding_model_name: str,
    replace_existing: bool,
) -> dict[str, object]:
    slug = _safe_slug(display_name)
    corpus_dir = USER_INDEX_ROOT / slug
    if corpus_dir.exists():
        if not replace_existing:
            raise ValueError(f"A custom corpus named '{display_name}' already exists.")
        shutil.rmtree(corpus_dir)

    docs_dir = corpus_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, object]] = []
    document_rows: list[dict[str, object]] = []
    for uploaded_file in uploaded_files:
        document = _read_uploaded_document(uploaded_file)
        (docs_dir / document.storage_name).write_text(document.text, encoding="utf-8")
        chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        document_rows.append(
            {
                "file": document.source_name,
                "stored_text_file": document.storage_name,
                "characters": len(document.text),
                "chunks": len(chunks),
            }
        )
        for chunk in chunks:
            metadata.append(
                {
                    "doc_path": document.source_name,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                }
            )

    if not metadata:
        raise ValueError("No chunks were produced. Check the uploaded files and chunk settings.")

    texts = [str(row["text"]) for row in metadata]
    embedder = load_embedder(embedding_model_name)
    vectors = embedder.encode(texts)
    index = build_faiss_index(vectors, normalize=EMBEDDING_NORMALIZE)

    index_path = corpus_dir / "index.faiss"
    meta_path = corpus_dir / "metadata.jsonl"
    manifest_path = corpus_dir / "manifest.json"
    save_faiss_index(index, index_path)
    save_metadata(metadata, meta_path)

    manifest = {
        "display_name": display_name,
        "slug": slug,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "document_count": len(document_rows),
        "chunk_count": len(metadata),
        "documents": document_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_retriever.clear()
    return manifest


def _preview_uploaded_chunks(uploaded_files, chunk_size: int, overlap: int, *, limit: int = 4) -> tuple[list[dict[str, object]], int]:
    previews: list[dict[str, object]] = []
    total_chunks = 0
    for uploaded_file in uploaded_files or []:
        try:
            document = _read_uploaded_document(uploaded_file)
        except ValueError:
            continue
        chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        total_chunks += len(chunks)
        for chunk in chunks:
            if len(previews) >= limit:
                continue
            previews.append(
                {
                    "file": document.source_name,
                    "chars": f"{chunk.start}-{chunk.end}",
                    "preview": _short_text(chunk.text, max_chars=220),
                }
            )
    return previews, total_chunks


def _custom_corpus_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, info in _list_custom_corpora().items():
        manifest = info.get("manifest") or {}
        rows.append(
            {
                "corpus": label.replace("Custom: ", ""),
                "documents": manifest.get("document_count", "-"),
                "chunks": manifest.get("chunk_count", "-"),
                "embedding_model": manifest.get("embedding_model", EMBEDDING_MODEL_NAME),
                "chunk_size": manifest.get("chunk_size", "-"),
                "overlap": manifest.get("overlap", "-"),
            }
        )
    return rows


def _analyze_uploaded_files(uploaded_files) -> dict[str, object] | None:
    if not uploaded_files:
        return None
    texts: list[str] = []
    document_count = 0
    for uploaded_file in uploaded_files:
        try:
            document = _read_uploaded_document(uploaded_file)
        except ValueError:
            continue
        texts.append(document.text)
        document_count += 1
    if not texts:
        return None

    joined = "\n\n".join(texts)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", joined) if part.strip()]
    non_empty_lines = [line.strip() for line in joined.splitlines() if line.strip()]
    avg_paragraph_chars = (
        sum(len(paragraph) for paragraph in paragraphs) / len(paragraphs)
        if paragraphs
        else len(joined)
    )
    avg_line_chars = (
        sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        if non_empty_lines
        else len(joined)
    )
    non_ascii_chars = sum(1 for char in joined if ord(char) > 127)
    newline_density = joined.count("\n") / max(1, len(joined))
    clause_markers = len(
        re.findall(
            r"\b(section|article|clause|agreement|policy|confidential|shall|whereas|definitions?)\b",
            joined,
            flags=re.IGNORECASE,
        )
    )
    list_markers = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)]|[a-zA-Z][.)])\s+", joined))
    return {
        "document_count": document_count,
        "total_characters": len(joined),
        "paragraph_count": len(paragraphs),
        "avg_paragraph_chars": round(avg_paragraph_chars, 1),
        "avg_line_chars": round(avg_line_chars, 1),
        "non_ascii_ratio": round(non_ascii_chars / max(1, len(joined)), 4),
        "newline_density": round(newline_density, 4),
        "clause_markers": clause_markers,
        "list_markers": list_markers,
    }


def _suggest_chunk_preset(stats: dict[str, object] | None) -> dict[str, object]:
    presets = {
        "Short notes / FAQ": {
            "chunk_size": 400,
            "overlap": 75,
            "reason": "Shorter chunks fit compact notes, FAQs, and short support-page passages.",
        },
        "General documents": {
            "chunk_size": 800,
            "overlap": 150,
            "reason": "Balanced setting for mixed prose documents.",
        },
        "Legal / policy documents": {
            "chunk_size": 1000,
            "overlap": 200,
            "reason": "Larger chunks preserve clauses, definitions, and policy sections across boundaries.",
        },
        "Long reports": {
            "chunk_size": 1200,
            "overlap": 250,
            "reason": "Longer chunks preserve section context in longer reports.",
        },
    }
    if not stats:
        return {"name": "General documents", **presets["General documents"]}

    total_chars = int(stats["total_characters"])
    avg_paragraph = float(stats["avg_paragraph_chars"])
    avg_line = float(stats["avg_line_chars"])
    newline_density = float(stats["newline_density"])
    clause_markers = int(stats["clause_markers"])
    list_markers = int(stats["list_markers"])

    if clause_markers >= 8 or (newline_density >= 0.004 and avg_line <= 180 and list_markers >= 5):
        name = "Legal / policy documents"
    elif total_chars >= 80_000 or avg_paragraph >= 900:
        name = "Long reports"
    elif total_chars <= 8_000 and avg_paragraph <= 350:
        name = "Short notes / FAQ"
    else:
        name = "General documents"
    return {"name": name, **presets[name]}


def _embedding_label_for_model(embedding_options: dict[str, str], model_name: str) -> str | None:
    for label, option_model_name in embedding_options.items():
        if option_model_name == model_name:
            return label
    return None


def _suggest_embedding_model(
    stats: dict[str, object] | None,
    embedding_options: dict[str, str],
) -> dict[str, str]:
    if not embedding_options:
        return {
            "label": EMBEDDING_MODEL_NAME,
            "model_name": EMBEDDING_MODEL_NAME,
            "reason": "Current project default.",
        }

    target_model = EMBEDDING_MODEL_NAME
    reason = "Balanced default for English documents and thesis demos."
    if stats:
        total_chars = int(stats["total_characters"])
        document_count = int(stats["document_count"])
        non_ascii_ratio = float(stats.get("non_ascii_ratio", 0.0))
        if non_ascii_ratio >= 0.025 and "BAAI/bge-m3" in embedding_options.values():
            target_model = "BAAI/bge-m3"
            reason = "Best fit when documents may include multilingual or mixed-script text."
        elif total_chars <= 10_000 and document_count <= 2 and "BAAI/bge-small-en-v1.5" in embedding_options.values():
            target_model = "BAAI/bge-small-en-v1.5"
            reason = "Fast indexing option for a small demonstration corpus."

    if target_model not in embedding_options.values():
        target_model = next(iter(embedding_options.values()))
        reason = "Selected from the embedding models currently available on this machine."

    label = _embedding_label_for_model(embedding_options, target_model) or target_model
    return {"label": label, "model_name": target_model, "reason": reason}


def _render_build_corpus_page() -> None:
    st.markdown('<div class="section-kicker">Build corpus</div>', unsafe_allow_html=True)
    st.subheader("Upload documents and build a local retrieval index")
    st.caption("This creates a private local FAISS index from uploaded text, Markdown, or PDF files.")

    if st.session_state.get("last_build_summary"):
        summary = st.session_state.pop("last_build_summary")
        st.success(
            f"Built '{summary['display_name']}' with {summary['document_count']} documents "
            f"and {summary['chunk_count']} chunks using {summary['embedding_model']}."
        )

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        corpus_name = st.text_input("Corpus name", value="my_local_corpus")
        uploaded_files = st.file_uploader(
            "Documents",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="Supports text, Markdown, and text-based PDFs. Scanned/image-only PDFs need OCR first.",
        )
    with right:
        stats = _analyze_uploaded_files(uploaded_files)

        st.markdown("**Embedding**")
        embedding_options = _available_embedding_model_options()
        embedding_suggestion = _suggest_embedding_model(stats, embedding_options)
        if stats:
            st.success(f"Suggested: {embedding_suggestion['label']}")
            st.caption(embedding_suggestion["reason"])
        else:
            st.caption("Upload files to get an embedding model suggestion.")
        use_embedding_suggestion = st.checkbox("Use suggested embedding model", value=True)
        embedding_model_values = list(embedding_options.values())
        default_embedding_model = (
            embedding_suggestion["model_name"] if use_embedding_suggestion else EMBEDDING_MODEL_NAME
        )
        if default_embedding_model not in embedding_model_values:
            default_embedding_model = embedding_model_values[0]
        default_embedding_index = embedding_model_values.index(default_embedding_model)
        embedding_label = st.selectbox(
            "Embedding model",
            list(embedding_options.keys()),
            index=default_embedding_index,
            help="The same embedding model will be saved with the corpus and reused for retrieval.",
        )
        embedding_model_name = embedding_options[embedding_label]
        st.caption(f"Model id: `{embedding_model_name}`")

        st.markdown("**Chunking**")
        suggestion = _suggest_chunk_preset(stats)
        if stats:
            st.success(
                f"Suggested: {suggestion['name']} "
                f"({suggestion['chunk_size']} chars, {suggestion['overlap']} overlap)"
            )
            st.caption(str(suggestion["reason"]))
            with st.expander("Document profile", expanded=False):
                st.write(
                    {
                        "documents": stats["document_count"],
                        "total_characters": stats["total_characters"],
                        "avg_paragraph_chars": stats["avg_paragraph_chars"],
                        "avg_line_chars": stats["avg_line_chars"],
                    }
                )
        else:
            st.caption("Upload files to get an automatic chunking suggestion.")

        use_suggestion = st.checkbox("Use suggested settings", value=True)
        default_chunk_size = int(suggestion["chunk_size"]) if use_suggestion else CHUNK_SIZE
        default_overlap = int(suggestion["overlap"]) if use_suggestion else CHUNK_OVERLAP
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=3000, value=default_chunk_size, step=100)
        overlap = st.number_input("Overlap", min_value=0, max_value=1000, value=default_overlap, step=25)
        replace_existing = st.checkbox("Replace corpus if name exists", value=True)

    if uploaded_files:
        previews, estimated_chunks = _preview_uploaded_chunks(uploaded_files, int(chunk_size), int(overlap))
        if estimated_chunks:
            st.caption(f"Estimated index size with these settings: {estimated_chunks} chunks.")
        if previews:
            with st.expander("Preview first chunks", expanded=False):
                st.dataframe(previews, hide_index=True, width="stretch")

    build_clicked = st.button("Build local index", type="primary", width="stretch")
    if build_clicked:
        if not uploaded_files:
            st.error("Upload at least one .txt, .md, or .pdf file.")
            return
        if overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
            return
        with st.spinner("Chunking, embedding, and writing the FAISS index..."):
            try:
                summary = _build_custom_corpus(
                    display_name=corpus_name,
                    uploaded_files=uploaded_files,
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                    embedding_model_name=embedding_model_name,
                    replace_existing=replace_existing,
                )
            except Exception as exc:  # Surface build failures clearly in the UI.
                st.error(str(exc))
                return
        st.session_state["last_build_summary"] = summary
        st.rerun()

    rows = _custom_corpus_rows()
    if rows:
        st.markdown("**Available custom corpora**")
        st.dataframe(rows, hide_index=True, width="stretch")


def _render_chunks(
    title: str,
    chunks: Iterable[RetrievedChunk],
    empty_text: str,
    *,
    key_prefix: str,
    default_limit: int = 8,
) -> None:
    chunks = list(chunks)
    st.markdown(f"**{title}**")
    if not chunks:
        st.caption(empty_text)
        return
    st.caption(f"{len(chunks)} chunks · {_context_char_count(chunks):,} context characters")
    show_all = len(chunks) <= default_limit or st.checkbox(
        f"Show all {len(chunks)} chunks",
        value=False,
        key=f"{key_prefix}_show_all_chunks",
    )
    visible_chunks = chunks if show_all else chunks[:default_limit]
    if not show_all:
        st.caption(f"Showing the first {len(visible_chunks)} ranked chunks.")
    for idx, chunk in enumerate(visible_chunks, start=1):
        with st.expander(_chunk_title(chunk, idx), expanded=False):
            st.write(_short_text(chunk.text))


def _render_run(run: DemoRun) -> None:
    st.subheader(run.label)
    _render_answer(run.answer)
    _render_run_badges(run)

    if not run.retrieved and not run.used:
        with st.expander("Run metrics", expanded=False):
            _render_run_metrics(run)
        return

    _render_status(run)
    _render_plain_correction_summary(run)
    with st.expander("Run metrics", expanded=False):
        _render_run_metrics(run)

    if run.routing_type or run.refined_queries or run.answer_selection_source:
        details = []
        if run.routing_type:
            details.append(f"Route: `{run.routing_type}`")
        if run.refined_queries:
            details.append("Rewrite: " + " | ".join(f"`{q}`" for q in run.refined_queries))
        if run.answer_selection_source:
            details.append(f"Answer selection: `{run.answer_selection_source}`")
        st.caption(" · ".join(details))

    safe_label = "".join(ch if ch.isalnum() else "_" for ch in run.label.lower())
    tab_answer_evidence, tab_candidates, tab_pipeline = st.tabs(
        ["Evidence sent to model", "Retrieved evidence", "Retrieval trace"]
    )
    with tab_answer_evidence:
        _render_source_summary(run.used, title="Source summary for final prompt context")
        _render_chunks(
            "Chunks sent to the model",
            run.used,
            "Local model mode does not use retrieved document chunks.",
            key_prefix=f"{safe_label}_used",
            default_limit=10,
        )
    with tab_candidates:
        _render_source_summary(run.retrieved, title="Source summary for candidate set")
        _render_chunks(
            "Retrieved candidates before final context selection",
            run.retrieved,
            "No retrieval was performed in model-only mode.",
            key_prefix=f"{safe_label}_retrieved",
            default_limit=10,
        )
    with tab_pipeline:
        st.markdown("**Retrieval and generation summary**")
        st.write(
            {
                "live_runtime_s": round(run.latency_s, 4),
                "retrieval_latency_s": round(run.retrieval_latency_s, 4),
                "generation_latency_s": round(run.generation_latency_s, 4),
                "candidate_chunks_before_context_limit": len(run.retrieved),
                "final_context_chunks": len(run.used),
                "final_context_chars": _context_char_count(run.used),
                "retrieved_context_tokens": run.retrieved_context_tokens,
                "prompt_tokens": run.prompt_tokens,
                "answer_tokens": run.answer_tokens,
            }
        )
        if run.first_pass_answer or run.corrected_pass_answer:
            with st.expander("HotpotQA answer-selection candidates", expanded=False):
                st.markdown("**First-pass answer**")
                st.write(run.first_pass_answer or "_None_")
                st.markdown("**Corrected-pass answer**")
                st.write(run.corrected_pass_answer or "_None_")
        if run.iteration_trace:
            st.markdown("**Correction trace**")
            for entry in run.iteration_trace:
                title = (
                    f"Iteration {entry.get('iteration_index')} · "
                    f"{entry.get('controller_decision')} · "
                    f"{entry.get('stop_reason')}"
                )
                with st.expander(title, expanded=False):
                    st.write(
                        {
                            "retrieval_query": entry.get("retrieval_query"),
                            "refined_query": entry.get("refined_query"),
                            "missing_aspect": entry.get("missing_aspect"),
                            "corrective_action": entry.get("corrective_action"),
                            "corrective_route": entry.get("corrective_route"),
                            "route_reason": entry.get("route_reason"),
                            "route_target_doc": entry.get("route_target_doc"),
                            "context_chars": entry.get("context_chars"),
                        }
                    )
        elif run.label == "Bounded Corrective RAG":
            st.caption("No trace was returned. This usually means correction stopped before a traced retry.")


def _render_run_answer_only(run: DemoRun) -> None:
    st.subheader(run.label)
    _render_answer(run.answer)
    _render_run_badges(run)


def _available_model_options() -> dict[str, Path]:
    available = {name: path for name, path in MODEL_OPTIONS.items() if path.exists()}
    return available or MODEL_OPTIONS


def _render_run_comparison(runs: list[DemoRun]) -> None:
    rows = []
    for run in runs:
        correction = "not applicable"
        if run.label == "Bounded Corrective RAG":
            correction = "activated" if run.iterations > 1 else "not activated"
        rows.append(
            {
                "pipeline": run.label,
                "answer_preview": _short_text(run.answer, max_chars=180),
                "runtime_s": round(run.latency_s, 2),
                "retrieval_calls": run.retrieval_calls,
                "final_chunks": len(run.used),
                "final_chars": _context_char_count(run.used),
                "correction": correction,
                "main_sources": _source_names(run.used),
            }
        )

    st.markdown("**Run comparison**")
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_grounded_runs(runs: list[DemoRun]) -> None:
    st.header("Document-grounded assistant")
    if not runs:
        st.caption("No grounded pipeline was selected.")
        return
    if len(runs) == 1:
        _render_run(runs[0])
        return

    _render_run_comparison(runs)
    tabs = st.tabs([run.label for run in runs])
    for tab, run in zip(tabs, runs):
        with tab:
            _render_run_answer_only(run)


def _render_results(*, grounded_runs: list[DemoRun], model_only_run: DemoRun | None) -> None:
    if model_only_run is not None:
        st.header("Local model answer")
        _render_run(model_only_run)
    else:
        _render_grounded_runs(grounded_runs)


def _clear_last_results() -> None:
    st.session_state.pop("last_grounded_runs", None)
    st.session_state.pop("last_model_only_run", None)
    st.session_state.pop("last_result_question", None)
    st.session_state.pop("last_result_mode", None)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _apply_page_style()

    st.sidebar.header("Configuration")
    workspace = st.sidebar.radio("Workspace", ["Ask", "Build corpus"], index=0)
    if workspace == "Build corpus":
        _render_build_corpus_page()
        return

    assistant_mode = st.sidebar.radio(
        "Assistant mode",
        ["Ask documents", "Ask local model"],
        index=0,
        on_change=_clear_last_results,
    )

    corpus_registry = _corpus_registry()
    corpus_label = ""
    selected_corpus_info: dict[str, object] | None = None
    pipeline_mode = "Compare traditional vs corrective"
    top_k = 5
    if assistant_mode == "Ask documents":
        corpus_label = st.sidebar.selectbox(
            "Corpus",
            list(corpus_registry.keys()),
            on_change=_clear_last_results,
        )
        selected_corpus_info = corpus_registry[corpus_label]
        st.sidebar.caption(f"Corpus embedding: `{selected_corpus_info.get('embedding_model') or EMBEDDING_MODEL_NAME}`")
        pipeline_mode = st.sidebar.radio(
            "Grounded pipeline",
            ["Traditional RAG", "Bounded Corrective RAG", "Compare traditional vs corrective"],
            index=2,
            on_change=_clear_last_results,
        )
        top_k = st.sidebar.slider(
            "Initial retrieval top-k",
            min_value=1,
            max_value=20,
            value=5,
            on_change=_clear_last_results,
        )
        with st.sidebar.expander("Retrieval settings", expanded=False):
            st.write(
                {
                    "initial_top_k": top_k,
                    "corrective_top_k_per_query": AGENTIC_CORRECTIVE_TOP_K,
                    "max_retrieval_control_iterations": AGENTIC_MAX_ITERS,
                    "final_context_budget_chars": CONTEXT_CHAR_BUDGET,
                }
            )
            st.caption(
                "Corrective mode may issue a primary rewrite retrieval and an optional backup retrieval. "
                "The displayed candidate set can therefore be larger than 20, while the final prompt context "
                "is still limited by the character budget."
            )

    available_models = _available_model_options()
    model_name = st.sidebar.selectbox(
        "Local model",
        list(available_models.keys()),
        on_change=_clear_last_results,
    )
    model_path = available_models[model_name]

    st.title(APP_TITLE)
    if assistant_mode == "Ask documents":
        st.caption(f"{corpus_label} · {pipeline_mode} · {model_name}")
    else:
        st.caption(f"General local model · {model_name}")

    examples = EXAMPLE_QUESTIONS.get(corpus_label) if assistant_mode == "Ask documents" else None
    if assistant_mode == "Ask documents" and examples:
        example_label = st.sidebar.selectbox("Example question", list(examples.keys()))
        example_question = examples[example_label]
        if st.session_state.get("selected_example_question") != example_question:
            st.session_state["selected_example_question"] = example_question
            st.session_state["question"] = example_question
        if st.sidebar.button("Use selected example"):
            st.session_state["selected_example_question"] = example_question
            st.session_state["question"] = example_question
            st.session_state.pop("last_grounded_runs", None)
            st.session_state.pop("last_model_only_run", None)
    elif assistant_mode == "Ask documents":
        st.sidebar.caption("Custom corpus selected. Enter your own question.")
        st.session_state.setdefault("question", "")
    else:
        st.session_state.setdefault("question", "")
    st.markdown('<div class="section-kicker">Ask</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Question",
        key="question",
        height=105,
        label_visibility="collapsed",
        placeholder=(
            "Ask a question against the selected local corpus..."
            if assistant_mode == "Ask documents"
            else "Ask the local model any general question..."
        ),
        on_change=_clear_last_results,
    )

    run_col, clear_col = st.columns([1, 1])
    run_clicked = run_col.button("Ask", type="primary", width="stretch")
    if clear_col.button("Clear results", width="stretch"):
        _clear_last_results()

    if run_clicked:
        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
            return
        if not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Loading local model..." if assistant_mode == "Ask local model" else "Loading local model and retriever..."):
            model = load_model(str(model_path))
            retriever = None
            if assistant_mode == "Ask documents":
                if selected_corpus_info is None:
                    st.error("Please select a corpus.")
                    return
                retriever = load_retriever(
                    str(selected_corpus_info["index_path"]),
                    str(selected_corpus_info["meta_path"]),
                    str(selected_corpus_info.get("embedding_model") or EMBEDDING_MODEL_NAME),
                )

        grounded_runs: list[DemoRun] = []
        model_only_run: DemoRun | None = None
        if assistant_mode == "Ask local model":
            with st.spinner("Running local model..."):
                model_only_run = _run_model_only(model, question)

        if assistant_mode == "Ask documents" and pipeline_mode in {"Traditional RAG", "Compare traditional vs corrective"}:
            with st.spinner("Running traditional RAG..."):
                assert retriever is not None
                grounded_runs.append(
                    _run_traditional(
                        corpus_label=corpus_label,
                        model=model,
                        retriever=retriever,
                        question=question,
                        top_k=top_k,
                    )
                )

        if assistant_mode == "Ask documents" and pipeline_mode in {"Bounded Corrective RAG", "Compare traditional vs corrective"}:
            with st.spinner("Running bounded corrective RAG... this can be slow locally."):
                assert retriever is not None
                grounded_runs.append(
                    _run_corrective(
                        corpus_label=corpus_label,
                        model=model,
                        retriever=retriever,
                        question=question,
                        top_k=top_k,
                    )
                )

        st.session_state["last_grounded_runs"] = grounded_runs
        st.session_state["last_model_only_run"] = model_only_run
        st.session_state["last_result_question"] = question.strip()
        st.session_state["last_result_mode"] = assistant_mode

    last_grounded_runs = st.session_state.get("last_grounded_runs")
    last_model_only_run = st.session_state.get("last_model_only_run")
    last_result_question = st.session_state.get("last_result_question")
    last_result_mode = st.session_state.get("last_result_mode")
    if (
        last_result_question == question.strip()
        and last_result_mode == assistant_mode
        and (last_grounded_runs is not None or last_model_only_run is not None)
    ):
        _render_results(
            grounded_runs=last_grounded_runs or [],
            model_only_run=last_model_only_run,
        )


if __name__ == "__main__":
    main()
