import re


FALLBACK = "I could not find the answer in the provided documents."

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "when", "where", "why", "how", "many", "much", "does", "do", "did",
    "can", "could", "should", "would", "will", "shall", "of", "for", "to",
    "in", "on", "at", "by", "from", "and", "or", "with", "company",
    "employee", "employees", "policy", "policies", "information", "please",
    "tell", "me", "about", "be", "it", "this", "that", "their", "they",
    "each", "every", "per", "day", "days", "leave",
    "entitled", "entitlement", "receive", "receives", "allowed",
}

SYNONYMS = {
    "vacation": {"annual", "leave"},
    "vacations": {"annual", "leave"},
    "remote": {"remote", "work"},
}


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS
    }


def specific_query_terms(question: str) -> set[str]:
    terms = tokens(question)
    expanded = set(terms)
    for term in terms:
        expanded.update(SYNONYMS.get(term, set()))
    return expanded


def context_supports_question(question: str, context: str) -> bool:
    query_terms = specific_query_terms(question)
    context_terms = tokens(context)
    return bool(query_terms and query_terms.intersection(context_terms))


def split_multi_part_question(question: str) -> list[str]:
    """Split simple multi-part questions while preserving question intent."""
    parts = re.split(r"\s+\band\b\s+", question.strip(), flags=re.IGNORECASE)
    if len(parts) == 1:
        return [question.strip()]

    first = parts[0].strip(" ?.")
    match = re.match(
        r"^(what|which|how|when|who)\b(?:\s+(?:is|are|many|much|can|does|do|should|would|will))?(?:\s+(?:the|a|an))?",
        first,
        flags=re.IGNORECASE,
    )
    prefix = match.group(0).strip() if match else ""
    normalized = [first + "?"]

    for part in parts[1:]:
        part = part.strip(" ?.")
        if re.match(r"^(what|which|how|when|who)\b", part, flags=re.IGNORECASE):
            normalized.append(part + "?")
        elif prefix:
            normalized.append(prefix + " " + part + "?")
        else:
            normalized.append(part + "?")

    return normalized


def sentence_list(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def focused_context(question: str, context: str) -> str:
    terms = specific_query_terms(question)
    sentences = sentence_list(context)
    matched = [sentence for sentence in sentences if terms.intersection(tokens(sentence))]
    return " ".join(dict.fromkeys(matched)) if matched else context
