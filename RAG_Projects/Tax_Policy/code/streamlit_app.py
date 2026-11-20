import os
from pathlib import Path

import streamlit as st
from rag_pipeline import get_rag_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

st.set_page_config(page_title="Policy & Tax RAG Assistant", page_icon="📑", layout="wide")

st.title("📑 Regulatory & Tax Policy RAG Assistant")
st.caption("Grounded Q&A with page-level citations powered by ChromaDB, BGE Embeddings & Groq")

# Sidebar Configuration
with st.sidebar:
    st.header("Retrieval Settings")
    top_k = st.slider("Number of Chunks to Retrieve (Top-K)", min_value=2, max_value=8, value=4)
    
    st.divider()
    st.subheader("Indexed Documents")
    if DATA_PATH.is_dir():
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
        if pdf_files:
            for f in pdf_files:
                st.markdown(f"- `{f}`")
        else:
            st.warning("No PDFs found in `Data/` folder.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Retrieved Evidence & Citations"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"**{idx}. {src['file']} (Page {src['page']})** — *Relevance Score: {src['score']}*")
                    st.info(src["snippet"])

# Handle User Input
if prompt := st.chat_input("Ask a question about your policy documents..."):
    if not CHROMA_PATH.is_dir():
        st.error("Vector database not found! Run `python code\\ingest.py` from the project folder first.")
        st.stop()

    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and verifying citations..."):
            answer, sources = get_rag_response(prompt, k=top_k)
            st.markdown(answer)
            
            if sources:
                with st.expander("View Retrieved Evidence & Citations"):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**{idx}. {src['file']} (Page {src['page']})** — *Relevance Score: {src['score']}*")
                        st.info(src["snippet"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })