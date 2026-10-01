from pathlib import Path

document_path = Path("documents/company_info.txt")

text = document_path.read_text()

print("Document loaded successfully!")
print(text)

# Split document into chunks
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
print("\nNumber of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(chunks)

print("\nEmbedding shape:", embeddings.shape)

import faiss
import numpy as np

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(np.array(embeddings).astype("float32"))

print("FAISS index created!")
print("Number of vectors:", index.ntotal)

query = "How many paid leave days do employees get?"

query_embedding = embedding_model.encode([query])

distances, indices = index.search(
    np.array(query_embedding).astype("float32"),
    k=2
)

print("\nQuery:", query)

for i in indices[0]:
    print("\nRelevant chunk:")
    print(chunks[i])

# Build context from retrieved chunks
retrieved_chunks = [chunks[i] for i in indices[0]]

context = "\n\n".join(retrieved_chunks)

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}

Answer:
"""

print("\n--- RAG Prompt ---")
print(prompt)

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

result = generator(
    prompt,
    max_new_tokens=30,
    do_sample=False
)

print("\n--- Final Answer ---")
print(result[0]["generated_text"])
