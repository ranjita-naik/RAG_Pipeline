from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from src.retrieval.pipeline import RetrievalPipeline
from src.config import settings

def build_rag_pipeline():
    """
    Builds a production-grade RAG pipeline:
    - Hybrid retrieval (BM25 + dense embeddings)
    - Optional reranking
    - Multi-stage retrieval pipeline
    """

    # LLM for answer generation
    llm = ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=0
    )

    # Build retrieval pipeline
    retrieval_pipeline = RetrievalPipeline(
        mode="hybrid",   # options: "dense", "hybrid"
        rerank=True      # toggle reranking
    )

    # Wrap retrieval pipeline so LangChain can use it
    class CustomRetriever:
        def get_relevant_documents(self, query):
            return retrieval_pipeline.retrieve(query)

    retriever = CustomRetriever()

    # Build RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return chain

