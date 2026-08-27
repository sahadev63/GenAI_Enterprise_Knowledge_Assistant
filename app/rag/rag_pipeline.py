import re

from app.ingestion.embedding import generate_embedding
from app.retrieval.vector_store import search_documents
from app.generation.prompt import create_rag_prompt
from app.generation.llm import generate_answer
from app.config import RAG_TOP_K, RAG_DISTANCE_THRESHOLD


FALLBACK = "I could not find the answer in the provided documents."

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "when", "where", "why", "how", "many", "much", "does", "do", "did",
    "can", "could", "should", "would", "will", "shall", "of", "for", "to",
    "in", "on", "at", "by", "from", "and", "or", "with", "company",
    "employee", "employees", "policy", "policies", "information", "please",
    "tell", "me", "about", "be", "it", "this", "that", "their", "they",
    "each", "every", "per", "day", "days", "leave"
}

SYNONYMS = {
    "vacation": {"annual", "leave"},
    "vacations": {"annual", "leave"},
    "remote": {"remote", "work"},
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS
    }


def _specific_query_terms(question: str) -> set[str]:
    terms = _tokens(question)
    expanded = set(terms)

    for term in terms:
        expanded.update(SYNONYMS.get(term, set()))

    return expanded


def _context_supports_question(question: str, context: str) -> bool:
    """
    Prevent generic words such as 'leave' from making an unrelated question
    appear answerable.
    """
    query_terms = _specific_query_terms(question)
    context_terms = _tokens(context)

    return bool(query_terms and query_terms.intersection(context_terms))


def _split_multi_part_question(question: str) -> list[str]:
    """
    Split simple multi-part questions so one missing topic cannot hide a
    different topic that is present in the knowledge base.
    """
    parts = re.split(r"\s+\band\b\s+", question.strip(), flags=re.IGNORECASE)

    if len(parts) == 1:
        return [question.strip()]

    first = parts[0].strip(" ?.")
    match = re.match(
        r"^(what|which|how|when|who)\b(?:\s+(?:is|are|many|much|can|does|do|"
        r"should|would|will))?",
        first,
        flags=re.IGNORECASE,
    )
    prefix = match.group(0).strip() if match else ""

    normalized = [first + "?"]

    for part in parts[1:]:
        part = part.strip(" ?.")

        # If the second clause already contains its own question words,
        # preserve it. Otherwise reuse the first clause's question intent.
        if re.match(r"^(what|which|how|when|who)\b", part, flags=re.IGNORECASE):
            normalized.append(part + "?")
        elif prefix:
            normalized.append(prefix + " " + part + "?")
        else:
            normalized.append(part + "?")

    return normalized


def _sentence_list(context: str) -> list[str]:
    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        context,
    )
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _focused_context(question: str, context: str) -> str:
    terms = _specific_query_terms(question)
    sentences = _sentence_list(context)

    matched = [
        sentence
        for sentence in sentences
        if terms.intersection(_tokens(sentence))
    ]

    return " ".join(dict.fromkeys(matched)) if matched else context


def _clean_generated_answer(answer: str, question: str) -> str:
    """
    Keep only generated sentences that contain a meaningful concept from the
    question. This prevents the small local model from echoing unrelated
    retrieved leave-policy sentences.
    """
    answer = answer.strip()

    if not answer:
        return ""

    if answer.lower().strip() == question.lower().strip():
        return ""

    if answer.lower().strip() == FALLBACK.lower():
        return ""

    unsupported_phrases = (
        "could not find",
        "not found in the provided documents",
        "not available",
        "does not contain",
        "do not contain",
    )
    if any(phrase in answer.lower() for phrase in unsupported_phrases):
        return ""

    generic_phrases = (
        "i'm ready to assist",
        "i am ready to assist",
        "what is your question",
    )
    if any(phrase in answer.lower() for phrase in generic_phrases):
        return ""

    terms = _specific_query_terms(question)

    relevant = [
        sentence
        for sentence in _sentence_list(answer)
        if terms.intersection(_tokens(sentence))
    ]

    return " ".join(dict.fromkeys(relevant)).strip()


def _answer_single_question(
    question: str,
    n_results: int,
    distance_threshold: float | None,
) -> str:
    query_embedding = generate_embedding(question)

    results = search_documents(
        query_embedding=query_embedding,
        n_results=n_results,
        distance_threshold=distance_threshold,
    )

    documents = results["documents"][0]
    unique_documents = list(dict.fromkeys(documents))

    if not unique_documents:
        return FALLBACK

    context = "\n\n".join(unique_documents)

    # Semantic similarity alone can match the generic word "leave" for
    # maternity/sick/laptop questions. Require a meaningful concept from the
    # question to occur in the retrieved context as a second grounding gate.
    if not _context_supports_question(question, context):
        return FALLBACK

    focused_context = _focused_context(question, context)

    prompt = create_rag_prompt(
        question=question,
        context=focused_context,
    )

    answer = generate_answer(prompt)
    cleaned = _clean_generated_answer(answer, question)

    if cleaned:
        return cleaned

    # If the local LLM ignores the instruction, return only the retrieved
    # evidence relevant to the question. This guarantees a grounded answer.
    evidence = [
        sentence
        for sentence in _sentence_list(focused_context)
        if _specific_query_terms(question).intersection(_tokens(sentence))
    ]

    return " ".join(dict.fromkeys(evidence)) if evidence else FALLBACK


def _topic_from_question(question: str) -> str:
    topic = re.sub(
        r"^(what|which|how|when|who)\b",
        "",
        question.strip(),
        flags=re.IGNORECASE,
    )
    topic = re.sub(
        r"\b(is|are|do|does|can|could|many|much)\b",
        "",
        topic,
        flags=re.IGNORECASE,
    )
    topic = re.sub(r"\\b(the|company|company's)\\b", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"\s+", " ", topic).strip(" ?.")
    return topic or "that information"


def answer_question(
    question: str,
    n_results: int = RAG_TOP_K,
    distance_threshold: float | None = RAG_DISTANCE_THRESHOLD,
) -> str:
    """
    Retrieve grounded context and generate an answer.

    Multi-part questions are answered independently, so available information
    is preserved even when another requested part is missing.
    """
    if not question or not question.strip():
        return FALLBACK

    parts = _split_multi_part_question(question)

    answers = []
    for index, part in enumerate(parts):
        answer = _answer_single_question(
            part,
            n_results=n_results,
            distance_threshold=distance_threshold,
        )

        # For multi-part questions, identify the missing part explicitly
        # instead of using the all-or-nothing fallback.
        if len(parts) > 1 and answer == FALLBACK:
            topic = _topic_from_question(part)
            answer = (
                f"Information about {topic} was not found in the "
                "provided documents."
            )

        answers.append(answer)

    final_parts = []
    for answer in answers:
        if answer and answer not in final_parts:
            final_parts.append(answer)

    return "\n\n".join(final_parts) if final_parts else FALLBACK
