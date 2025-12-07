import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

VECTOR_DB_PATH = "vectorstore"

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

def build_rag_pipeline():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",   # simplest, deterministic
        return_source_documents=True
    )
    return qa

