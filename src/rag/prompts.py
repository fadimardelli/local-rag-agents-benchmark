SYSTEM_PROMPT = (
    "You are a careful legal assistant. "
    "Answer the question using only the provided context. "
    "If the context is insufficient, say you don't know."
)

USER_TEMPLATE = (
    "Context:\n"
    "{context}\n\n"
    "Question:\n"
    "{question}\n\n"
    "Answer:"
)

LABEL_SYSTEM_PROMPT = (
    "You are a legal NLI classifier. "
    "Given the context and the hypothesis, output exactly one label: "
    "Entailment, Contradiction, or NotMentioned. "
    "Do not output anything else."
)

HOTPOT_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Return only the final short answer text. "
    "Do not add explanation, prefix, or extra words."
)

CONTROLLER_DECISION_SYSTEM_PROMPT = (
    "You are a retrieval evaluator for a local RAG system. "
    "Judge only whether the current retrieval is good enough to answer the question. "
    "Be strict about retrieval quality and source consistency. "
    "If the question refers to a specific agreement, party, or document, treat wrong-source retrieval as insufficient. "
    "Return only STOP or RETRY."
)

CONTROLLER_DECISION_TEMPLATE = (
    "Question:\n"
    "{question}\n\n"
    "Retrieved Context:\n"
    "{context}\n\n"
    "Decide whether the retrieved context is relevant enough to answer the question.\n"
    "Return STOP only if the retrieval is both relevant and source-consistent.\n"
    "Otherwise return RETRY.\n"
    "Return only one word.\n"
)

CONTROLLER_REWRITE_SYSTEM_PROMPT = (
    "You rewrite questions into short retrieval queries for a local RAG system. "
    "Preserve the user's intent. Preserve any agreement names, party names, dates, and distinctive entities from the original question. "
    "Never broaden the scope to generic document categories. "
    "Do not use Boolean syntax, parentheses, OR, AND, quotes, or explanations. "
    "Return only the rewritten query text."
)

CONTROLLER_REWRITE_TEMPLATE = (
    "Original Question:\n"
    "{question}\n\n"
    "Entity Hints To Preserve:\n"
    "{entity_hints}\n\n"
    "Retrieved Context:\n"
    "{context}\n\n"
    "Write one short retrieval query that stays faithful to the original question, keeps the important entities, "
    "and targets what the current retrieval appears to be missing.\n"
    "Return only the rewritten query text.\n"
)

QUERY_TRANSFORM_SYSTEM_PROMPT = (
    "You convert a user question into retrieval-oriented contract language for dense retrieval. "
    "Write a short hypothetical contract clause or answer passage that would likely appear in a relevant document. "
    "Prefer formal legal wording such as party roles, obligations, conditions, and scope terms when supported by the question. "
    "Preserve important entities from the question. "
    "Keep party roles consistent with the question. "
    "Do not invent section numbers, deadlines, or unsupported factual details. "
    "Do not explain what you are doing. "
    "Return only the hypothetical passage."
)

QUERY_TRANSFORM_TEMPLATE = (
    "User Question:\n"
    "{question}\n\n"
    "Write a concise hypothetical contract passage that would directly answer this question if it appeared in the source document.\n"
    "Use likely clause language, not conversational phrasing.\n"
    "Return only the passage.\n"
)

MULTI_ACTION_CONTROLLER_SYSTEM_PROMPT = (
    "You are a retrieval controller for a local RAG workflow. "
    "Choose the next action that is most likely to improve retrieval quality. "
    "Prefer SEARCH_CHUNKS when a better dense retrieval pass is needed, "
    "SELECT_DOCUMENT only when one retrieved document is clearly better supported than the others and the workflow should focus on it, "
    "REFORMULATE_QUERY when the wording should change, "
    "EXPAND_DOCUMENT when the right document seems present but the wrong chunk is shown, "
    "treat explicit agreement names, party names, and source-consistency as important retrieval signals, "
    "for generic clause hypotheses without distinctive source cues, avoid committing to one document unless several top chunks support that same document, "
    "avoid expanding chunks from a different-looking document title when the question names a specific agreement or parties, "
    "prefer STOP when the local document frontier already seems exhausted and further expansion is unlikely to add new evidence, "
    "and STOP only when the current retrieval is sufficient to answer the question. "
    "Return valid JSON only."
)

MULTI_ACTION_CONTROLLER_TEMPLATE = (
    "Question:\n"
    "{question}\n\n"
    "Current Query:\n"
    "{current_query}\n\n"
    "Current Focused Document:\n"
    "{focused_document}\n\n"
    "Available Actions:\n"
    "{available_actions}\n\n"
    "Prior Actions:\n"
    "{action_history}\n\n"
    "Retrieved Context:\n"
    "{retrieved_context}\n\n"
    "Choose exactly one action from the available actions listed above.\n"
    "Use the source-match notes when judging whether a chunk likely comes from the right document.\n"
    "Use SELECT_DOCUMENT only when one document is clearly supported by the retrieval and the workflow has not explicitly focused on it yet.\n"
    "If the query is a generic clause statement and several different documents look similar, prefer SEARCH_CHUNKS or REFORMULATE_QUERY over early document commitment.\n"
    "If local exploration of the current document already appears exhausted, prefer STOP over another expansion.\n"
    "Use EXPAND_DOCUMENT when the document looks right but the local chunk seems incomplete or adjacent evidence is likely needed.\n"
    "If you choose REFORMULATE_QUERY or SEARCH_CHUNKS, set \"query\" to the next query text.\n"
    "If you choose SELECT_DOCUMENT, set \"document_id\" to a 1-based document number from the retrieved context.\n"
    "If you choose EXPAND_DOCUMENT, optionally set \"anchor_chunk\" to a 1-based chunk number from the retrieved context.\n"
    "Return JSON with keys: action, query, document_id, anchor_chunk, rationale.\n"
    "Use an empty string for query when not needed.\n"
    "Do not include any text before or after the JSON object.\n"
)


def build_user_prompt(context: str, question: str) -> str:
    return USER_TEMPLATE.format(context=context, question=question)


def build_controller_prompt(context: str, question: str) -> str:
    return CONTROLLER_DECISION_TEMPLATE.format(context=context, question=question)


def build_rewrite_prompt(question: str, entity_hints: str, context: str) -> str:
    return CONTROLLER_REWRITE_TEMPLATE.format(
        question=question,
        entity_hints=entity_hints,
        context=context,
    )


def build_query_transform_prompt(question: str) -> str:
    return QUERY_TRANSFORM_TEMPLATE.format(question=question)


def build_multi_action_controller_prompt(
    *,
    question: str,
    current_query: str,
    focused_document: str,
    available_actions: str,
    action_history: str,
    retrieved_context: str,
) -> str:
    return MULTI_ACTION_CONTROLLER_TEMPLATE.format(
        question=question,
        current_query=current_query,
        focused_document=focused_document,
        available_actions=available_actions,
        action_history=action_history,
        retrieved_context=retrieved_context,
    )
