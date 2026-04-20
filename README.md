# Agentic RAG — FAISS + HuggingFace + Ollama

A fully local Agentic RAG system. No paid APIs, no cloud dependencies.

```
User Query
    │
    ▼
┌─────────────┐
│   Router    │  LLM decides the path
└─────────────┘
   │        │         │
   ▼        ▼         ▼
Vector DB  Web Search  Direct Answer
(FAISS)  (DuckDuckGo)
   │        │
   └────────┘
        │
        ▼
  Relevance Grader   ← drops irrelevant chunks
        │
        ▼
   Generator         ← LLM answers from context
        │
        ▼
 Hallucination Check ← verifies answer is grounded
        │
        ▼
   Final Answer
```

## Stack

| Component  | Tool                              | Cost  |
|------------|-----------------------------------|-------|
| LLM        | Ollama — llama3.2                 | Free  |
| Embeddings | HuggingFace — all-MiniLM-L6-v2   | Free  |
| Vector DB  | FAISS (flat / hnsw / hnsw_pq)    | Free  |
| Web Search | DuckDuckGo                        | Free  |

## Setup

### 1. Install Ollama + pull the model
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your documents
Drop `.pdf` or `.txt` files into the `data/` folder.

### 4. Index — choose your FAISS index type

```bash
# Exact search — best accuracy (default, good for small datasets)
python main.py --index

# HNSW — fast graph-based ANN (good for medium/large datasets)
python main.py --index --index-type hnsw

# HNSW + Product Quantization — compressed RAM (good for huge datasets)
python main.py --index --index-type hnsw_pq
```

### 5. Ask questions

```bash
# Single query
python main.py --query "What is the refund policy?"

# Interactive chat
python main.py --chat
```

## FAISS Index Types Explained

### flat — Exact Search
Guarantees the true nearest neighbors every time.

### hnsw — Hierarchical Navigable Small World
Builds a multi-layer graph. Query navigates the graph greedily.
Near-exact results, much faster than flat on large datasets.

### hnsw_pq — HNSW + Product Quantization
Splits each 384-dim vector into 8 sub-vectors of 48 dims,
quantizes each to 1 byte → 8 bytes/vector instead of 1536 bytes.
HNSW graph handles fast navigation, PQ handles compression.

## Example queries the router handles

| Query                                     | Routed to     |
|-------------------------------------------|---------------|
| "What is 2 + 2?"                          | direct        |
| "Who is the CEO of Apple?"                | web_search    |
| "What does the privacy policy say?"       | vectorstore   |
