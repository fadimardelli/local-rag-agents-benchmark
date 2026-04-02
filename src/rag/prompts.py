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
    "Be strict about retrieval quality, clause specificity, and source consistency, but do not request another search if the needed evidence is already present. "
    "If the question refers to a specific agreement, party, or document, treat wrong-source retrieval as insufficient. "
    "If the chunks are from the right document and already contain the likely answer span, clause heading, date, defined term, obligation, exception, covenant, or location cue needed to answer, return STOP. "
    "Do not ask for a retry just because a cleaner or more precise chunk might exist. "
    "Use RETRY only when the current chunks are wrong-source, mixed-source in a harmful way, or clearly missing the specific fact, clause, condition, exception, covenant, term, or answer span needed. "
    "Return only STOP or RETRY."
)

CONTROLLER_DECISION_TEMPLATE = (
    "Question:\n"
    "{question}\n\n"
    "Retrieved Context:\n"
    "{context}\n\n"
    "Decide whether the retrieved context is sufficient to answer the question.\n"
    "Return STOP if the retrieval is relevant, source-consistent, and already contains enough evidence to answer.\n"
    "If the likely answer span or clause is already present, return STOP even if another search might find a cleaner snippet.\n"
    "Otherwise return RETRY.\n"
    "Return only one word.\n"
)

CONTROLLER_REWRITE_SYSTEM_PROMPT = (
    "You are a legal retrieval strategist for a local RAG system. "
    "Your job is to improve retrieval when the current chunks are insufficient. "
    "First identify the single most important fact missing from the retrieved context that prevents answering the question. "
    "Then write one targeted retrieval query for that missing fact. "
    "If useful, you may provide one backup query, but only if it targets the same missing fact from a different legal phrasing. "
    "Preserve important entities from the original question, including agreement names, party names, jurisdictions, dates, and distinctive terms. "
    "Preserve polarity and negation. "
    "Use compact retrieval-friendly legal wording such as likely clause terms, obligations, exceptions, conditions, and scope terms. "
    "Prefer short keyword-style search queries, not full questions or sentences. "
    "Keep each query under 18 words. "
    "Do not invent facts, section numbers, or entities not present in the question. "
    "Do not broaden to generic legal topics. "
    "Return JSON only."
)

CONTROLLER_REWRITE_TEMPLATE = (
    "Original Question:\n"
    "{question}\n\n"
    "Entity Hints To Preserve:\n"
    "{entity_hints}\n\n"
    "Retrieved Context:\n"
    "{context}\n\n"
    "Identify the most important missing fact needed to answer the question.\n"
    "Write compact keyword-style retrieval queries, not natural-language questions.\n\n"
    "Return JSON with this schema:\n"
    "{{\n"
    '  "missing_fact": "<short phrase>",\n'
    '  "primary_query": "<short keyword query>",\n'
    '  "backup_query": "<optional short keyword query or NONE>"\n'
    "}}\n"
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
