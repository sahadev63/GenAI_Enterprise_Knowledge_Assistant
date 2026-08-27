from app.generation.llm import generate_answer
from app.generation.prompt import create_rag_prompt


question = "What is RAG?"

context = """
RAG stands for Retrieval-Augmented Generation.
It combines information retrieval with a language model.
The retrieval system finds relevant information from documents,
and the language model uses that information to generate an answer.
"""

prompt = create_rag_prompt(
    question=question,
    context=context
)

answer = generate_answer(prompt)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)