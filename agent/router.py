from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


LLM_MODEL = "llama3.2"


# Simple, clear prompt — the LLM must reply with ONLY one word
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query router for a RAG system.
Your job is to decide where to look for the answer.

Reply with ONLY one of these three words — nothing else:
- vectorstore   (if the question is about documents the user uploaded)
- web_search    (if the question needs current news, live data, or general web knowledge)
- direct        (if you can answer this directly without any documents, e.g. math, definitions)
"""),
    ("human", "Query: {query}")
])


def route_query(query):
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = ROUTER_PROMPT | llm

    response = chain.invoke({"query": query})
    decision = response.content.strip().lower()

    # Normalize — sometimes the LLM adds punctuation or extra words
    if "vectorstore" in decision:
        return "vectorstore"
    elif "web" in decision:
        return "web_search"
    else:
        return "direct"
