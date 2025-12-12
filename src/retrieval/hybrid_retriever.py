from langchain_community.retrievers import BM25Retriever
from src.ingestion.vectorstore import load_vectorstore
from src.retrieval.retriever_base import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, docs, vector_k=4, bm25_k=6, alpha=0.5):
        """
        alpha = weight for dense vs lexical results.
        alpha=0.5 means equal fusion.
        """
        self.bm25 = BM25Retriever.from_documents(docs)
        self.vectorstore = load_vectorstore()
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.alpha = alpha

    def retrieve(self, query: str):
        # lexical results
        bm25_docs = self.bm25.get_relevant_documents(query)[:self.bm25_k]

        # dense vector results
        dense_docs = self.vectorstore.similarity_search(query, k=self.vector_k)

        # build a score dictionary
        scores = {}

        for rank, d in enumerate(bm25_docs):
            scores[id(d)] = scores.get(id(d), 0) + (1 - rank / len(bm25_docs)) * (1 - self.alpha)

        for rank, d in enumerate(dense_docs):
            scores[id(d)] = scores.get(id(d), 0) + (1 - rank / len(dense_docs)) * self.alpha

        # merge, sort by fused score
        combined = list({id(d): d for d in bm25_docs + dense_docs}.values())
        combined_sorted = sorted(combined, key=lambda d: scores[id(d)], reverse=True)

        return combined_sorted

