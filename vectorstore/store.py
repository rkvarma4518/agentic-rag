import os
import pickle
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


FAISS_DIR      = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM      = 384

HNSW_M         = 32
HNSW_EF_SEARCH = 64
HNSW_EF_BUILD  = 200

PQ_M           = 8
PQ_NBITS       = 8


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_flat_index():
    return faiss.IndexFlatIP(EMBED_DIM)


def build_hnsw_index():
    index = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M)
    index.hnsw.efSearch       = HNSW_EF_SEARCH
    index.hnsw.efConstruction = HNSW_EF_BUILD
    return index


def build_hnsw_pq_index():
    index = faiss.IndexHNSWPQ(EMBED_DIM, PQ_M, HNSW_M, PQ_NBITS)
    index.hnsw.efSearch       = HNSW_EF_SEARCH
    index.hnsw.efConstruction = HNSW_EF_BUILD
    return index


def build_vectorstore(chunks, index_type="flat"):
    os.makedirs(FAISS_DIR, exist_ok=True)

    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  Index type      : {index_type.upper()}")
    print(f"  Chunks to embed : {len(chunks)}")

    embeddings = get_embeddings()
    texts      = [c.page_content for c in chunks]

    print("  Embedding chunks (runs locally on CPU)...")
    vectors    = embeddings.embed_documents(texts)
    vectors_np = np.array(vectors, dtype=np.float32)

    if index_type == "hnsw":
        raw_index = build_hnsw_index()

    elif index_type == "hnsw_pq":
        raw_index = build_hnsw_pq_index()
        if not raw_index.is_trained:
            print(f"  Training PQ quantizer on {len(vectors_np)} vectors...")
            raw_index.train(vectors_np)

    else:
        raw_index  = build_flat_index()
        index_type = "flat"

    raw_index.add(vectors_np)
    print(f"  Added {raw_index.ntotal} vectors to index.")

    vectorstore       = FAISS.from_documents(chunks, embeddings)
    vectorstore.index = raw_index

    vectorstore.save_local(FAISS_DIR)

    meta = {"index_type": index_type, "embed_model": EMBED_MODEL, "dim": EMBED_DIM}
    with open(os.path.join(FAISS_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"  Index saved to  : {FAISS_DIR}/")
    _print_index_summary(index_type, raw_index)
    return vectorstore


def load_vectorstore():
    index_path = os.path.join(FAISS_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return None

    meta_path = os.path.join(FAISS_DIR, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        print(f"  Index type  : {meta['index_type'].upper()}")
        print(f"  Embed model : {meta['embed_model']}")

    embeddings  = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"  Total vectors : {vectorstore.index.ntotal}")
    return vectorstore


def get_retriever(vectorstore, top_k=4):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def _print_index_summary(index_type, index):
    print("\n  ── Index summary ──────────────────────────────")
    print(f"  Type       : {index_type.upper()}")
    print(f"  Vectors    : {index.ntotal}")
    print(f"  Dimensions : {EMBED_DIM}")

    if index_type == "hnsw":
        print(f"  HNSW M     : {HNSW_M}")
        print(f"  efSearch   : {HNSW_EF_SEARCH}")
        print(f"  efBuild    : {HNSW_EF_BUILD}")

    elif index_type == "hnsw_pq":
        print(f"  HNSW M     : {HNSW_M}")
        print(f"  PQ subvecs : {PQ_M}")
        print(f"  PQ bits    : {PQ_NBITS}")
        orig_kb = index.ntotal * EMBED_DIM * 4 / 1024
        pq_kb   = index.ntotal * PQ_M * (PQ_NBITS // 8) / 1024
        if pq_kb > 0:
            ratio = orig_kb / pq_kb
            print(f"  Compression: ~{ratio:.0f}x  ({orig_kb:.1f} KB → {pq_kb:.1f} KB for vectors)")

    print("  ───────────────────────────────────────────────\n")