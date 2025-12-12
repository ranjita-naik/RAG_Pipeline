from src.rag.chain_builder import build_rag_chain
import streamlit as st

st.title("📚 Production RAG System")

@st.cache_resource
def load_chain():
    return build_rag_chain()

qa = load_chain()

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        result = qa(query)

    st.subheader("Answer")
    st.write(result["result"])

    st.subheader("Sources")
    for i, doc in enumerate(result["source_documents"]):
        st.markdown(f"**Source {i+1}:**")
        st.write(doc.page_content[:600] + "…")

