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

## Project Structure

```
agentic_rag/
├── main.py                       # entry point — all CLI commands here
├── requirements.txt
├── data/                         # drop your PDF / TXT docs here
│   └── sample_company_docs.txt
├── vectorstore/
│   └── store.py                  # FAISS index + HuggingFace embeddings
├── agent/
│   ├── router.py                 # routes query: vectorstore / web / direct
│   ├── retriever.py              # fetches chunks from FAISS
│   ├── grader.py                 # filters irrelevant chunks
│   ├── generator.py              # generates answer from context
│   └── hallucination.py          # checks answer is grounded in sources
└── utils/
    ├── document_loader.py        # loads + chunks PDFs and TXT files
    └── web_search.py             # free DuckDuckGo search
```

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
Brute-force comparison against every vector.
Guarantees the true nearest neighbors every time.
- Best for: < 10k chunks
- RAM: `n * 384 * 4 bytes` (no compression)
- Accuracy: 100%

### hnsw — Hierarchical Navigable Small World
Builds a multi-layer graph. Query navigates the graph greedily.
Near-exact results, much faster than flat on large datasets.
- Best for: 10k – 1M chunks
- RAM: slightly more than flat (graph edges stored)
- Accuracy: ~97–99%
- Key params: `M` (connectivity), `efSearch` (recall/speed tradeoff)

### hnsw_pq — HNSW + Product Quantization
Splits each 384-dim vector into 8 sub-vectors of 48 dims,
quantizes each to 1 byte → 8 bytes/vector instead of 1536 bytes.
HNSW graph handles fast navigation, PQ handles compression.
- Best for: 1M+ chunks, or low-RAM machines
- RAM: ~192x smaller vector storage
- Accuracy: ~95–97% (2–5% loss vs exact)
- Needs training data: at least ~256 vectors

## Example queries the router handles

| Query                                     | Routed to     |
|-------------------------------------------|---------------|
| "What is our refund policy?"              | vectorstore   |
| "Latest news about LLMs?"                | web_search    |
| "What is 2 + 2?"                          | direct        |
| "Who is the CEO of Apple?"                | web_search    |
| "What does the privacy policy say?"       | vectorstore   |
