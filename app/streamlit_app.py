import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))       
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                    

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_chain import build_rag_pipeline  # this should now work

import streamlit as st

st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("📚 RAG Chatbot Demo")
st.caption("Ask questions based on your two RAG research papers.")

@st.cache_resource
def get_qa_chain():
    return build_rag_pipeline()

qa = get_qa_chain()

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Retrieving information..."):
        result = qa(query)

    st.subheader("Answer")
    st.write(result["result"])

    with st.expander("Retrieved Context"):
        for i, doc in enumerate(result["source_documents"]):
            st.write(f"📄 **Source {i+1}:**")
            st.write(doc.page_content[:800] + "...")

