from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


LLM_MODEL = "llama3.2"


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the user's question using ONLY
the context provided below. Be concise and factual.

If the context does not contain enough information to answer,
say "I don't have enough information to answer this question."

Do NOT make up information that is not in the context.

Context:
{context}
"""),
    ("human", "Question: {query}")
])


DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question concisely."),
    ("human", "{query}")
])


def generate_answer(query, context):
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = RAG_PROMPT | llm

    response = chain.invoke({"query": query, "context": context})
    return response.content.strip()


def generate_direct_answer(query):
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = DIRECT_PROMPT | llm

    response = chain.invoke({"query": query})
    return response.content.strip()
