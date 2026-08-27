from ollama import chat


MODEL_NAME = "llama3.2:3b"


def generate_answer(prompt: str) -> str:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict enterprise knowledge assistant. "
                    "Use only the supplied context. "
                    "Do not invent facts or add unrelated information."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"].strip()
