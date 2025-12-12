
# 📚 **RAG Pipeline – Production-Ready Retrieval-Augmented Generation System**

This repository contains a **production-grade Retrieval-Augmented Generation (RAG) system** built with:

- **Hybrid Retrieval** (BM25 + dense embeddings)
- **LLM-based Reranking** for improved context relevance
- **Configurable Chunking & Metadata Preservation**
- **FAISS Vector Indexing**
- **FastAPI Backend** for easy deployment
- **Streamlit Frontend** for interactive exploration
---

# 🔥 **Key Features**

### ✅ **1. Modular, Scalable Architecture**
The system is split into clean modules:

```
src/
  ingestion/     → loaders, chunkers, embedding, vector index builder
  retrieval/     → dense, hybrid, reranking, retrieval pipeline
  rag_chain.py   → builds the final RAG pipeline
  api/           → FastAPI service
app/
  streamlit_app.py → user-friendly chat interface
```

Supports:
- Multi-stage retrieval
- Reranking
- Configurable vector stores
- Clean orchestration logic

---

### ✅ **2. Hybrid Retrieval**
Combines:

- **BM25 Lexical Retriever**
- **Dense Vector Retrieval (FAISS)**
- **Weighted score fusion (alpha blending)**

Benefits:
- Better recall
- Better robustness to keyword mismatch
- Stronger hallucination resistance

---

### ✅ **3. LLM-Based Reranking**
A lightweight reranker boosts precision using an LLM to score documents for the query.

Results in:
- More relevant chunks
- Fewer hallucinations
- Higher answer faithfulness

---

### ✅ **4. Configurable Chunking & Metadata**
Chunking preserves metadata such as:
- Start index
- Document IDs
- PDF page numbers

This enables:
- Reranking
- Citation-based answers
- Precise traceability

---

### ✅ **5. FastAPI Backend (Production Deployment)**
Expose the RAG pipeline as a REST API:

- `/ask` endpoint returns answer + sources
- Stateless and deployable on Docker, serverless, or VM
- Easy integration into existing products

---

### ✅ **6. Streamlit Chat UI (Optional Frontend)**
A simple web interface for:
- Asking questions
- Viewing retrieved context
- Testing pipeline variants (dense, hybrid, reranked)

---

# 🧱 **Project Structure**

```
.
├── src/
│   ├── config.py
│   ├── rag_chain.py
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vectorstore.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── retriever_base.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── pipeline.py
│   │
│   ├── api/
│   │   └── fastapi_app.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── *.pdf
├── vectorstore/
└── README.md
```

---

# ⚙️ **Installation**

### 1. Clone the repository

```bash
git clone https://github.com/ranjita-naik/RAG_Pipeline.git
cd RAG_Pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your_key_here"
```

---

# 🏗️ **Building the Vector Index**

Place your PDFs in the `data/` folder.

Then run:

```bash
python -m src.ingestion.pipeline
```

This will:

- Load all PDFs
- Split into chunks
- Generate embeddings
- Build a FAISS vector index in `vectorstore/`

---

# 🤖 **Running the RAG API (FastAPI)**

```bash
uvicorn src.api.fastapi_app:app --reload --port 8000
```

Make a request:

```bash
curl -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{"question": "What is retrieval-augmented generation?"}'
```

---

# 💬 **Running the Streamlit App**

```bash
streamlit run app/streamlit_app.py
```

---

# 🧠 **How Retrieval Works**

### **Stage 1: Lexical Search (BM25)**
Captures keyword-based relevance.

### **Stage 2: Dense Retrieval**
FAISS vector search using embeddings.

### **Stage 3: Score Fusion**
Blends BM25 + dense rankings:

```
final_score = α * dense + (1 - α) * bm25
```

### **Stage 4: LLM Reranking (Optional)**
Reorders top candidates using LLM scoring.

### **Stage 5: Context Assembly & Answer Generation**
Passes best documents into a deterministic LLM for answer generation.


