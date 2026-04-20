from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


LLM_MODEL = "llama3.2"


GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader.
Given a user question and a document chunk, decide if the chunk is relevant.

Reply with ONLY 'yes' or 'no'.
- yes: the chunk contains information useful for answering the question
- no: the chunk is off-topic or does not help answer the question
"""),
    ("human", "Question: {query}\n\nDocument chunk:\n{document}")
])


def grade_documents(query, docs):
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = GRADER_PROMPT | llm

    relevant_docs = []

    for doc in docs:
        response = chain.invoke({
            "query": query,
            "document": doc.page_content
        })
        score = response.content.strip().lower()

        if "yes" in score:
            relevant_docs.append(doc)

    print(f"  Grader kept {len(relevant_docs)} / {len(docs)} chunks as relevant.")
    all_irrelevant = len(relevant_docs) == 0
    return relevant_docs, all_irrelevant
