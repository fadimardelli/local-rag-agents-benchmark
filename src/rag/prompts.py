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


def build_user_prompt(context: str, question: str) -> str:
    return USER_TEMPLATE.format(context=context, question=question)
