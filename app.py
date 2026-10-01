import streamlit as st

from rag_pipeline import build_vector_store, get_relevant_chunks
from generate import answer_question

st.set_page_config(page_title="Chat With Your Docs")

st.title("📄 Chat With Your Docs")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file is not None:
    with open("temp_doc.txt", "wb") as f:
        f.write(uploaded_file.read())

    if "vector_store" not in st.session_state:
        with st.spinner("Indexing document..."):
            st.session_state.vector_store = build_vector_store("temp_doc.txt")

        st.success("Document indexed! Ask away.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Ask a question about your document"):
        st.chat_message("user").write(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Thinking..."):
            chunks = get_relevant_chunks(
                st.session_state.vector_store,
                question
            )

            answer = answer_question(question, chunks)

        st.chat_message("assistant").write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })