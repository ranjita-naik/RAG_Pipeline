from src.ingestion.vectorstore import load_vectorstore
from .retriever_base import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(self, k=4):
        self.vs = load_vectorstore()
        self.retriever = self.vs.as_retriever(search_kwargs={"k": k})

    def retrieve(self, query: str):
        return self.retriever.get_relevant_documents(query)
