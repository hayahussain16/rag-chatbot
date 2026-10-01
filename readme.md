# Chat With Your Docs - RAG Chatbot

## Problem Statement

Build a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a text document and ask questions about its content.

The system retrieves relevant information from the document and uses an LLM to generate a grounded answer.

## Architecture

Documents
↓
Text Chunks
↓
Sentence-Transformer Embeddings
↓
FAISS Vector Store
↓
User Question
↓
Similarity Search
↓
Top-K Relevant Chunks
↓
Prompt Context
↓
FLAN-T5 LLM
↓
Grounded Answer

## Technologies Used

- Python
- LangChain
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- FLAN-T5
- Streamlit

## How to Run

### 1. Install dependencies

```bash
pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu streamlit


