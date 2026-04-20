import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_documents():
    documents = []

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data/ folder. Add your PDFs or TXT files there.")
        return []

    files = os.listdir(DATA_DIR)
    if not files:
        print("No files found in data/. Add PDFs or TXT files to index.")
        return []

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)

        if filename.endswith(".pdf"):
            print(f"  Loading PDF: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

        elif filename.endswith(".txt"):
            print(f"  Loading TXT: {filename}")
            loader = TextLoader(filepath)
            documents.extend(loader.load())

    print(f"  Loaded {len(documents)} pages/sections from {len(files)} file(s).")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks.")
    return chunks
