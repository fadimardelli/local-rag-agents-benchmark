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

ANSWER_SELECTION_SYSTEM_PROMPT = (
    "You are a careful answer selector for open-domain question answering. "
    "You will receive a question, candidate answer A with its supporting context, and candidate answer B with its supporting context. "
    "Choose the candidate that is more directly and specifically supported by its context. "
    "Prefer exact supported answers over vague, partial, or hallucinated ones. "
    "If one candidate is unknown and the other is directly supported, choose the supported one. "
    "If neither candidate is directly supported, choose unknown. "
    "Return JSON only."
)

ANSWER_SELECTION_TEMPLATE = (
    "Question:\n"
    "{question}\n\n"
    "Candidate A Answer:\n"
    "{answer_a}\n\n"
    "Candidate A Context:\n"
    "{context_a}\n\n"
    "Candidate B Answer:\n"
    "{answer_b}\n\n"
    "Candidate B Context:\n"
    "{context_b}\n\n"
    "Return JSON with this schema:\n"
    "{{\n"
    '  "choice": "<A or B or UNKNOWN>",\n'
    '  "reason": "<short phrase>"\n'
    "}}\n"
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

CONTROLLER_GLOBAL_DECISION_SYSTEM_PROMPT = (
    "You are a retrieval evaluator for a local RAG system handling open-domain or multi-document questions. "
    "Judge only whether the current retrieval already contains enough evidence to answer the question. "
    "Be strict about evidence completeness, but do not request another search if the current chunks already provide the needed entities, bridge facts, or comparisons. "
    "Return STOP when the retrieved context already supports the answer with enough connected evidence, even if a cleaner second search might exist. "
    "Return RETRY only when the retrieval looks partial, one-sided, or likely misses a needed bridge fact, comparison target, or second source. "
    "Return only STOP or RETRY."
)

CONTROLLER_GLOBAL_DECISION_TEMPLATE = (
    "Question:\n"
    "{question}\n\n"
    "Retrieved Context:\n"
    "{context}\n\n"
    "Decide whether the retrieved context already contains enough connected evidence to answer the question.\n"
    "Return STOP if the current context already includes the needed entities, bridge facts, comparisons, or supporting sources.\n"
    "Return RETRY only if the evidence still looks partial, one-sided, or missing a likely second source or bridge fact.\n"
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

CONTROLLER_BRIDGE_REWRITE_SYSTEM_PROMPT = (
    "You are a retrieval strategist for a multi-document local RAG system. "
    "The first retrieval found only part of what is needed. "
    "Identify the single most important missing bridge fact, second entity, relation, or comparison target needed to answer the question. "
    "Then write one targeted global retrieval query for that missing piece of evidence. "
    "If useful, provide one backup query that targets the same missing bridge from a different phrasing. "
    "Preserve important entities, polarity, and comparison structure from the original question. "
    "Prefer compact keyword-style search queries, not full sentences. "
    "Keep each query under 18 words. "
    "Do not invent facts or unsupported entities. "
    "Return JSON only."
)

CONTROLLER_BRIDGE_REWRITE_TEMPLATE = (
    "Original Question:\n"
    "{question}\n\n"
    "Entity Hints To Preserve:\n"
    "{entity_hints}\n\n"
    "Retrieved Context:\n"
    "{context}\n\n"
    "Identify the most important missing bridge fact or second evidence target still needed.\n"
    "Write compact keyword-style retrieval queries for a second global search.\n\n"
    "Return JSON with this schema:\n"
    "{{\n"
    '  "missing_fact": "<short phrase>",\n'
    '  "primary_query": "<short keyword query>",\n'
    '  "backup_query": "<optional short keyword query or NONE>"\n'
    "}}\n"
)

CONTROLLER_BRIDGE_PLAN_SYSTEM_PROMPT = (
    "You are a retrieval strategist for a local RAG system handling open-domain or multi-document questions. "
    "Your job is to decide whether a second global retrieval is actually needed. "
    "If the current retrieval already contains enough connected evidence to answer, do not request another search. "
    "If another retrieval is needed, step back from the original wording and identify the more general missing relation, role, location, office, time period, comparison attribute, or bridge concept that would connect the evidence. "
    "Write one primary step-back query for that missing abstract relation or bridge concept. "
    "If useful, write one secondary query for a concrete entity-focused follow-up that complements the first query rather than paraphrasing it. "
    "Preserve important entities, polarity, and comparison structure from the original question. "
    "Prefer compact keyword-style retrieval queries, not full sentences. "
    "Keep each query under 18 words. "
    "Do not invent unsupported facts or entities. "
    "Return JSON only."
)

CONTROLLER_BRIDGE_PLAN_TEMPLATE = (
    "Original Question:\n"
    "{question}\n\n"
    "Entity Hints To Preserve:\n"
    "{entity_hints}\n\n"
    "Retrieved Context:\n"
    "{context}\n\n"
    "Decide whether another global retrieval is needed.\n"
    "If not needed, set need_retry to NO and set both queries to NONE.\n"
    "If needed, set need_retry to YES and write a step-back query for the missing relation or bridge concept.\n"
    "Use backup_query only when it adds a complementary concrete follow-up, not just a rephrasing.\n\n"
    "Return JSON with this schema:\n"
    "{{\n"
    '  "need_retry": "<YES or NO>",\n'
    '  "missing_fact": "<short phrase describing the missing evidence>",\n'
    '  "primary_query": "<short keyword step-back query or NONE>",\n'
    '  "backup_query": "<optional short complementary follow-up query or NONE>"\n'
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


def build_global_controller_prompt(context: str, question: str) -> str:
    return CONTROLLER_GLOBAL_DECISION_TEMPLATE.format(context=context, question=question)


def build_rewrite_prompt(question: str, entity_hints: str, context: str) -> str:
    return CONTROLLER_REWRITE_TEMPLATE.format(
        question=question,
        entity_hints=entity_hints,
        context=context,
    )


def build_bridge_rewrite_prompt(question: str, entity_hints: str, context: str) -> str:
    return CONTROLLER_BRIDGE_REWRITE_TEMPLATE.format(
        question=question,
        entity_hints=entity_hints,
        context=context,
    )


def build_bridge_plan_prompt(question: str, entity_hints: str, context: str) -> str:
    return CONTROLLER_BRIDGE_PLAN_TEMPLATE.format(
        question=question,
        entity_hints=entity_hints,
        context=context,
    )


def build_query_transform_prompt(question: str) -> str:
    return QUERY_TRANSFORM_TEMPLATE.format(question=question)


def build_answer_selection_prompt(
    *,
    question: str,
    answer_a: str,
    context_a: str,
    answer_b: str,
    context_b: str,
) -> str:
    return ANSWER_SELECTION_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        context_a=context_a,
        answer_b=answer_b,
        context_b=context_b,
    )
