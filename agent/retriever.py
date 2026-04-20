def retrieve_documents(query, retriever):
    print(f"  Searching vector store for: '{query}'")
    docs = retriever.invoke(query)
    print(f"  Retrieved {len(docs)} chunks.")
    return docs


def format_docs(docs):
    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"[{i}] Source: {source}" + (f", Page {page}" if page else "")
        parts.append(f"{label}\n{doc.page_content}")

    return "\n\n".join(parts)
