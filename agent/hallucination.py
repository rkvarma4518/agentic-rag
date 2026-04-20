from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


LLM_MODEL = "llama3.2"


HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a fact-checker.
Given a context and a generated answer, decide if the answer is grounded in the context.

Reply with ONLY 'grounded' or 'hallucinated'.
- grounded:     every claim in the answer can be traced back to the context
- hallucinated: the answer contains claims not supported by the context
"""),
    ("human", "Context:\n{context}\n\nGenerated answer:\n{answer}")
])


def check_hallucination(answer, context):
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = HALLUCINATION_PROMPT | llm

    response = chain.invoke({"context": context, "answer": answer})
    result = response.content.strip().lower()

    is_grounded = "grounded" in result
    status = "✓ grounded" if is_grounded else "✗ hallucinated"
    print(f"  Hallucination check: {status}")
    return is_grounded
