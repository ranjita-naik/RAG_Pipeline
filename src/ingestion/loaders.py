import os
from langchain_community.document_loaders import PyPDFLoader
from src.config import settings

def load_pdfs():
    docs = []
    for file in os.listdir(settings.DATA_FOLDER):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(settings.DATA_FOLDER, file))
            docs.extend(loader.load())
    return docs

