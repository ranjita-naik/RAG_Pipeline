import os
from langchain_community.vectorstores import FAISS
from src.ingestion.embedder import get_embedder
from src.config import settings

def save_vectorstore(vectorstore):
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    vectorstore.save_local(settings.VECTOR_DB_PATH)

def load_vectorstore():
    embeddings = get_embedder()
    return FAISS.load_local(
        settings.VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=False
    )

