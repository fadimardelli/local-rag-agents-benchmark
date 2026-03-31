from src.rag.multi_action_agentic import (
    ACTION_EXPAND_DOCUMENT,
    ACTION_SEARCH_CHUNKS,
    ACTION_REFORMULATE_QUERY,
    ACTION_STOP,
    MultiActionDecision,
    MultiActionAgenticRAG,
)
from src.rag.retriever import RetrievedChunk, Retriever


def test_parse_controller_output_json_reformulation() -> None:
    rag = MultiActionAgenticRAG.__new__(MultiActionAgenticRAG)
    decision = rag._parse_controller_output(
        '{"action":"REFORMULATE_QUERY","query":"termination for convenience notice","anchor_chunk":2,"rationale":"Need clause wording"}',
        current_query="original question",
    )

    assert decision.action == ACTION_REFORMULATE_QUERY
    assert decision.query == "termination for convenience notice"
    assert decision.anchor_chunk_index == 2


def test_parse_controller_output_fallback_stop() -> None:
    rag = MultiActionAgenticRAG.__new__(MultiActionAgenticRAG)
    decision = rag._parse_controller_output("STOP", current_query="original question")

    assert decision.action == ACTION_STOP
    assert decision.query == "original question"


def test_parse_controller_output_null_query_falls_back_to_current() -> None:
    rag = MultiActionAgenticRAG.__new__(MultiActionAgenticRAG)
    decision = rag._parse_controller_output(
        '{"action":"SEARCH_CHUNKS","query":null,"anchor_chunk":null,"rationale":null}',
        current_query="original question",
    )

    assert decision.action == "SEARCH_CHUNKS"
    assert decision.query == "original question"
    assert decision.anchor_chunk_index is None
    assert decision.rationale == ""


def test_expand_document_returns_local_neighbors() -> None:
    retriever = Retriever.__new__(Retriever)
    retriever.metadata = [
        {"doc_path": "docA", "start": 0, "end": 100, "text": "chunk-1"},
        {"doc_path": "docA", "start": 100, "end": 200, "text": "chunk-2"},
        {"doc_path": "docA", "start": 200, "end": 300, "text": "chunk-3"},
        {"doc_path": "docA", "start": 300, "end": 400, "text": "chunk-4"},
        {"doc_path": "docB", "start": 0, "end": 100, "text": "other-doc"},
    ]
    retriever.doc_to_indices = {
        "docA": [0, 1, 2, 3],
        "docB": [4],
    }

    seed = RetrievedChunk(doc_path="docA", start=200, end=300, text="chunk-3", score=0.9)
    expanded = retriever.expand_document([seed], k=3, anchor=seed)

    assert [chunk.start for chunk in expanded] == [100, 300, 200]
    assert all(chunk.doc_path == "docA" for chunk in expanded)
    assert expanded[0].score >= expanded[-1].score


def test_parse_controller_output_json_expand() -> None:
    rag = MultiActionAgenticRAG.__new__(MultiActionAgenticRAG)
    decision = rag._parse_controller_output(
        '{"action":"EXPAND_DOCUMENT","query":"","anchor_chunk":1,"rationale":"Right contract, wrong local span"}',
        current_query="original question",
    )

    assert decision.action == ACTION_EXPAND_DOCUMENT
    assert decision.query == "original question"
    assert decision.anchor_chunk_index == 1


def test_source_guard_blocks_wrong_document_expansion() -> None:
    retrieved = [
        RetrievedChunk(
            doc_path="data/corpus/contractnli/CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt",
            start=0,
            end=100,
            text="",
            score=0.9,
        ),
        RetrievedChunk(
            doc_path="data/corpus/contractnli/NDA-M5-Systems.txt",
            start=0,
            end=100,
            text="",
            score=0.8,
        ),
    ]
    decision = MultiActionDecision(
        action=ACTION_EXPAND_DOCUMENT,
        query="original question",
        anchor_chunk_index=2,
        rationale="Looks relevant locally",
    )

    guarded, applied, anchor_score, best_score = MultiActionAgenticRAG._apply_source_guard(
        question="Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; Does the document indicate anything?",
        retrieved=retrieved,
        decision=decision,
    )

    assert applied is True
    assert guarded.action == ACTION_SEARCH_CHUNKS
    assert anchor_score < best_score


def test_doc_title_tokens_split_realistic_contract_filename() -> None:
    tokens = MultiActionAgenticRAG._doc_title_tokens(
        "data/corpus/contractnli/CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt"
    )

    assert "cop" in tokens
    assert "acc" in tokens
    assert "top" in tokens
    assert "mentors" in tokens


def test_expand_document_progresses_into_unseen_neighbors() -> None:
    retriever = Retriever.__new__(Retriever)
    retriever.metadata = [
        {"doc_path": "docA", "start": 0, "end": 100, "text": "chunk-1"},
        {"doc_path": "docA", "start": 100, "end": 200, "text": "chunk-2"},
        {"doc_path": "docA", "start": 200, "end": 300, "text": "chunk-3"},
        {"doc_path": "docA", "start": 300, "end": 400, "text": "chunk-4"},
        {"doc_path": "docA", "start": 400, "end": 500, "text": "chunk-5"},
        {"doc_path": "docA", "start": 500, "end": 600, "text": "chunk-6"},
    ]
    retriever.doc_to_indices = {"docA": [0, 1, 2, 3, 4, 5]}

    current_window = [
        RetrievedChunk(doc_path="docA", start=100, end=200, text="chunk-2", score=0.8),
        RetrievedChunk(doc_path="docA", start=200, end=300, text="chunk-3", score=0.9),
        RetrievedChunk(doc_path="docA", start=300, end=400, text="chunk-4", score=0.8),
    ]
    anchor = current_window[1]
    expanded = retriever.expand_document(
        current_window,
        k=4,
        anchor=anchor,
        covered_positions={1, 2, 3},
    )

    assert {chunk.start for chunk in expanded} == {0, 100, 200, 400}
