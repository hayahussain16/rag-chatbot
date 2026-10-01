from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")


def answer_question(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.
Context: {context}
Question: {question}
Answer:"""

    result = generator(prompt, max_length=200)
    return result[0]["generated_text"]