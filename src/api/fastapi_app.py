from fastapi import FastAPI
from pydantic import BaseModel
from src.rag.chain_builder import build_rag_chain

app = FastAPI(title="RAG API", version="1.0")

qa = build_rag_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    result = qa(query.question)
    return {
        "answer": result["result"],
        "sources": [
            {"content": d.page_content, "metadata": d.metadata}
            for d in result["source_documents"]
        ]
    }

