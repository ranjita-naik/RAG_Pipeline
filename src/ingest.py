import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_FOLDER = "data"
VECTOR_DB_PATH = "vectorstore"

def load_pdfs():
    docs = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
            docs.extend(loader.load())
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"Saved vectorstore to {VECTOR_DB_PATH}/")
    return vectorstore

def build_index():
    print("📄 Loading PDFs...")
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages from PDFs.")

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("🔍 Creating vector index...")
    create_vectorstore(chunks)

if __name__ == "__main__":
    build_index()

