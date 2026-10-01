from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_store(file_path: str):
    # 1. Load the raw document
    loader = TextLoader(file_path)
    documents = loader.load()

    # 2. Split into overlapping chunks so context isn't lost at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed each chunk with an open-source sentence-transformer model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Store embeddings in a FAISS vector index for fast similarity search
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def get_relevant_chunks(vector_store, query: str, k: int = 3):
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]