import os
import pickle
import tempfile

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from utils.embedder import get_embeddings, query_llm
from utils.extractor import extract_text_from_file

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

st.title("🧠 Ask the Docs - Upload and Ask")

uploaded_files = st.file_uploader("Upload documents (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    docs_path = os.path.join(VECTORSTORE_DIR, "docs.pkl")
    if os.path.exists(index_path): os.remove(index_path)
    if os.path.exists(docs_path): os.remove(docs_path)

    all_texts = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = extract_text_from_file(tmp_path)

        import textwrap
        chunks = textwrap.wrap(text, width=800, break_long_words=False)
        all_texts.extend(chunks)

    embeddings = get_embeddings(all_texts)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    with open(os.path.join(VECTORSTORE_DIR, "docs.pkl"), "wb") as f:
        pickle.dump(all_texts, f)
    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "index.faiss"))

    st.success("Documents processed and indexed ✅")

query = st.text_input("Ask a question")

if query:
    if not os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")):
        st.error("No documents indexed yet.")
    else:
        index = faiss.read_index(os.path.join(VECTORSTORE_DIR, "index.faiss"))
        with open(os.path.join(VECTORSTORE_DIR, "docs.pkl"), "rb") as f:
            all_texts = pickle.load(f)

        query_embed = get_embeddings([query])[0]
        D, I = index.search(np.array([query_embed]).astype("float32"), k=3)
        context_chunks = [all_texts[i] for i in I[0]]

        answer = query_llm(context_chunks, query)

        st.markdown("### 🧠 Answer")
        st.markdown(answer)

        st.markdown("### 📚 Context Used")
        for i, chunk in enumerate(context_chunks):
            st.markdown(f"**Context {i+1}:**")
            st.caption(chunk)