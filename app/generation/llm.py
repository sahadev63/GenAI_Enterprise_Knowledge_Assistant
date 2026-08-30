import json
import re

from ollama import chat


MODEL_NAME = "llama3.2:3b"
SYSTEM_PROMPT = (
    "You are a strict enterprise knowledge assistant. "
    "Use only the supplied context or evidence. "
    "Do not invent facts or add unrelated information."
)


def chat_text(prompt: str, system_prompt: str | None = None) -> str:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt or SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()


def generate_answer(prompt: str) -> str:
    return chat_text(prompt)


def generate_json(prompt: str) -> dict:
    """Generate and parse a JSON object from the local LLM."""
    raw = chat_text(
        prompt,
        system_prompt=(
            "You are an enterprise AI control agent. "
            "Follow the requested JSON schema exactly. "
            "Return JSON only. Do not add markdown or commentary."
        ),
    )

    cleaned = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Extract the first JSON object from a model response that contains
        # accidental leading/trailing text.
        start = cleaned.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(cleaned[start:])

    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object.")
    return parsed
