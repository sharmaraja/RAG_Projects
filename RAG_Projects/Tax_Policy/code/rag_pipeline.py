import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """
You are a precise Financial & Regulatory Policy Assistant.
Answer the user's question using ONLY the provided context snippets below.

Rules:
1. If the answer is not explicitly contained in the context, state: "I cannot find this information in the provided documents." Do NOT guess or use outside knowledge.
2. Cite sources inline after key claims using [filename, page N].
3. Keep the explanation structured, clear, and concise.

Context:
{context}
"""

def get_rag_response(query: str, k: int = 4):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )

    # Retrieve top-k chunks with relevance scores
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    
    if not results:
        return "No relevant documents found in the database.", []

    context_blocks = []
    sources = []

    for doc, score in results:
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", 0) + 1  # Convert 0-indexed to 1-indexed
        
        context_blocks.append(
            f"[Source: {source_file}, page {page_num}]\n{doc.page_content}"
        )
        sources.append({
            "file": source_file,
            "page": page_num,
            "score": round(score, 3),
            "snippet": doc.page_content
        })

    formatted_context = "\n\n---\n\n".join(context_blocks)

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.0  # Zero temperature minimizes hallucination
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": formatted_context,
        "question": query
    })

    return response.content, sources