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
