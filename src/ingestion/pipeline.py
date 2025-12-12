from src.ingestion.loaders import load_pdfs
from src.ingestion.chunker import split_documents
from src.ingestion.vectorstore import save_vectorstore
from src.ingestion.embedder import get_embedder
from langchain_community.vectorstores import FAISS

def build_index():
    print("📄 Loading PDFs...")
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages.")

    print("✂️ Splitting documents...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("🔍 Embedding chunks...")
    embedder = get_embedder()
    vectorstore = FAISS.from_documents(chunks, embedder)

    print("💾 Saving vectorstore...")
    save_vectorstore(vectorstore)

    print("✅ Index built successfully.")

