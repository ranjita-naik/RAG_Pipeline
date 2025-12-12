from typing import Literal
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.ingestion.loaders import load_pdfs
from src.ingestion.chunker import split_documents

class RetrievalPipeline:
    def __init__(
        self,
        mode: Literal["dense", "hybrid"] = "dense",
        rerank: bool = True,
    ):
        docs = load_pdfs()
        chunks = split_documents(docs)

        if mode == "dense":
            self.retriever = DenseRetriever(k=6)
        elif mode == "hybrid":
            self.retriever = HybridRetriever(chunks, vector_k=4, bm25_k=8, alpha=0.5)
        
        self.reranker = Reranker() if rerank else None

    def retrieve(self, query):
        docs = self.retriever.retrieve(query)
        if self.reranker:
            docs = self.reranker.rerank(query, docs)
        return docs

