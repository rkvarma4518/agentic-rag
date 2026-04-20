import argparse
from colorama import Fore, Style, init

from utils.document_loader import load_documents, split_documents
from vectorstore.store import build_vectorstore, load_vectorstore, get_retriever
from agent.router import route_query
from agent.retriever import retrieve_documents, format_docs
from agent.grader import grade_documents
from agent.generator import generate_answer, generate_direct_answer
from agent.hallucination import check_hallucination
from utils.web_search import search_web, format_web_results

init(autoreset=True)


def index_documents(index_type):
    print(Fore.CYAN + "\n📂 Loading documents...")
    docs = load_documents()

    if not docs:
        return

    print(Fore.CYAN + "\n✂  Splitting into chunks...")
    chunks = split_documents(docs)

    print(Fore.CYAN + f"\n🔢 Building FAISS index  [{index_type.upper()}] ...")
    build_vectorstore(chunks, index_type=index_type)

    print(Fore.GREEN + "\n✅ Indexing complete! You can now run queries.\n")


def run_query(query, retriever):
    print(Fore.YELLOW + f"\n🔍 Query: {query}")
    print("─" * 60)

    print(Fore.CYAN + "\n[1] Routing query...")
    route = route_query(query)
    print(f"    Decision → {Fore.MAGENTA}{route.upper()}")

    context = ""

    if route == "direct":
        print(Fore.CYAN + "\n[2] Answering directly from LLM knowledge...")
        answer = generate_direct_answer(query)
        print(Fore.GREEN + f"\n💬 Answer:\n{answer}\n")
        return answer

    if route == "vectorstore":
        print(Fore.CYAN + "\n[2] Retrieving from FAISS vector store...")

        if retriever is None:
            print(Fore.YELLOW + "    No vector store found — falling back to web search.")
            route = "web_search"
        else:
            docs = retrieve_documents(query, retriever)

            print(Fore.CYAN + "\n[3] Grading retrieved chunks for relevance...")
            relevant_docs, all_irrelevant = grade_documents(query, docs)

            if all_irrelevant:
                print(Fore.YELLOW + "    No relevant chunks found → escalating to web search.")
                route = "web_search"
            else:
                context = format_docs(relevant_docs)

    if route == "web_search":
        print(Fore.CYAN + "\n[2/3] Searching the web (DuckDuckGo)...")
        results = search_web(query)
        context = format_web_results(results)

    print(Fore.CYAN + "\n[4] Generating answer from context...")
    answer = generate_answer(query, context)

    print(Fore.CYAN + "\n[5] Checking hallucinations...")
    is_grounded = check_hallucination(answer, context)

    if not is_grounded:
        print(Fore.RED + "    ⚠  Answer may not be fully grounded in the retrieved sources.")
        answer += "\n\n⚠ Note: This answer may contain claims not directly supported by retrieved sources."

    print(Fore.GREEN + f"\n💬 Answer:\n{answer}\n")
    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG — FAISS + HuggingFace Embeddings + Ollama LLM",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--index",
        action="store_true",
        help="Index all documents in the data/ folder"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "hnsw", "hnsw_pq"],
        metavar="TYPE",
        help=(
            "FAISS index type to use when indexing (default: flat)\n"
            "  flat     — exact search, best accuracy, slow on large datasets\n"
            "  hnsw     — fast ANN graph search, high recall\n"
            "  hnsw_pq  — hnsw + product quantization (compressed vectors, saves RAM)"
        )
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a single question and exit"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive chat session"
    )

    args = parser.parse_args()

    if args.index:
        index_documents(args.index_type)
        return

    print(Fore.CYAN + "\n🔄 Loading vector store...")
    vectorstore = load_vectorstore()
    retriever   = get_retriever(vectorstore) if vectorstore else None

    if vectorstore is None:
        print(Fore.YELLOW + (
            "\n⚠  No vector store found.\n"
            "   Web search and direct answers still work.\n"
            "   To index documents: python main.py --index\n"
        ))

    if args.query:
        run_query(args.query, retriever)

    elif args.chat:
        print(Fore.CYAN + "\n🤖 Agentic RAG Chat  (type 'exit' to quit)\n")
        while True:
            try:
                query = input(Fore.WHITE + "You: ").strip()
                if query.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                if not query:
                    continue
                run_query(query, retriever)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    else:
        parser.print_help()


if __name__ == "__main__":
    main()