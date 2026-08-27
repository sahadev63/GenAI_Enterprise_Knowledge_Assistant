from app.ingestion.embedding import generate_embedding
from app.retrieval.vector_store import search_documents
from app.generation.prompt import create_rag_prompt
from app.generation.llm import generate_answer

question = "How many annual leave days do employees get and what is the sick-leave entitlement?"

query_embedding = generate_embedding(question)


results = search_documents(
    query_embedding=query_embedding,
    n_results=3,
    distance_threshold=1.0,
)

documents = results["documents"][0]

# Remove duplicate documents while preserving order
unique_documents = list(dict.fromkeys(documents))

context = "\n\n".join(unique_documents)

prompt = create_rag_prompt(
    question=question,
    context=context,
)

answer = generate_answer(prompt)

print("Retrieved documents:")
print(results["documents"])

print()
print("Metadata:")
print(results["metadatas"])

print()
print("Distances:")
print(results["distances"])

print()
print("Context:")
print(context)

print()
print("RAG Prompt:")
print(prompt)

print()
print("Final Answer:")
print(answer)