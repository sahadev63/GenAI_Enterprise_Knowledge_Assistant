from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-base"


# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using the provided context and question.
    """

    prompt = f"""
You are an enterprise knowledge assistant.

You must answer ONLY from the CONTEXT.

IMPORTANT RULES:
1. If the CONTEXT directly contains the answer, answer the QUESTION.
2. If the CONTEXT does not contain the answer, respond exactly:
I could not find the answer in the provided documents.
3. Do not use your own knowledge.
4. Do not infer or guess.
5. Do not use information that is not present in the CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()