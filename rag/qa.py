from __future__ import annotations

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag.config import settings
from rag.ingest import get_vectorstore


SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided context.
Rules:
- If the answer is not clearly contained in the context, say you don't know based on the provided documents.
- Cite sources for key statements.
- Keep answers concise and factual.
"""

@dataclass
class SourceHit:
    source: str # Filename/URL
    page: int | None # Page number (None for non-PDFs)
    snippet: str  # First 280 chars of the chunk
    score: float  # Relevance score

def rewrite_question_with_context(question: str, chat_history: list[dict]) -> str:
    """
    Rewrite a follow-up question into a standalone question using recent chat history.
    Keeps retrieval clean while enabling conversational follow-ups.
    """
    if not chat_history:
        return question

    # Only last few turns to avoid token bloat
    recent = chat_history[-6:]

    history_text = []
    for m in recent:
        role = m.get("role", "").upper()
        content = m.get("content", "")
        history_text.append(f"{role}: {content}")

    prompt = (
        "You are given a conversation and a follow-up question.\n"
        "Rewrite the follow-up question so it is fully self-contained.\n\n"
        f"Conversation:\n{chr(10).join(history_text)}\n\n"
        f"Follow-up question:\n{question}\n\n"
        "Standalone question:"
    )

    llm = ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key, temperature=0)
    resp = llm.invoke(prompt)
    return (resp.content or "").strip() or question

def _format_context(docs: List[Document]) -> str:
    """
    Formats retrieved documents into numbered blocks for the LLM.
    
    Each document chunk is formatted with:
    - A numbered citation [1], [2], etc.
    - Source filename/URL and page number (if applicable)
    - The full document content
    
    Args:
        docs: List of Document objects from vector search
    
    Returns:
        str: Formatted context string with numbered citations separated by '---'
    
    Example output:
        [1] SOURCE: python_guide.pdf, page 5
        Python was created by Guido van Rossum in 1991...
        
        ---
        
        [2] SOURCE: programming_basics.txt
        Python is a high-level language known for readability...
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        page_str = f", page {page + 1}" if isinstance(page, int) else ""
        blocks.append(f"[{i}] SOURCE: {src}{page_str}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def _make_sources(docs_with_scores: List[Tuple[Document, float]]) -> List[SourceHit]:
    """
    Converts retrieved documents into SourceHit objects with metadata.
    
    Extracts source information from vector search results and creates
    structured objects for UI display. Page numbers are converted from
    0-indexed to 1-indexed for human readability.
    
    Args:
        docs_with_scores: List of tuples containing (Document, distance_score)
                         where lower scores mean higher similarity
    
    Returns:
        List[SourceHit]: List of SourceHit objects containing:
            - source: filename or URL
            - page: 1-indexed page number (None for non-PDFs)
            - snippet: first 280 characters with '…' if truncated
            - score: distance score from vector search
    """
    hits: List[SourceHit] = []
    for d, score in docs_with_scores:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        snippet = (d.page_content[:280] + "…") if len(d.page_content) > 280 else d.page_content
        hits.append(SourceHit(source=src, page=(page + 1) if isinstance(page, int) else None, snippet=snippet, score=float(score)))
    return hits


def ask_question(
    question: str,
    strict: bool = True, # strict: Refuse to answer if relevance is too low
    min_relevance: float = 0.35, # min_relevance: Threshold for strict mode (0.0 - 0.9)
    k: int = 6, # k: Number of document chunks to retrieve (2-12)
    use_conversation_memory: bool = False, # use_conversation_memory: Enable follow-up questions with context
    chat_history: list[dict] | None = None, # chat_history: List of chat messages for conversation memory
) -> Dict[str, Any]:
    """
    strict + min_relevance:
      We use similarity_search_with_score where LOWER score means "closer" for Chroma cosine distance.
      We'll convert to a pseudo "relevance" as 1 - distance, then threshold.
    """
    if not question.strip():
        return {"answer": "Ask a question.", "sources": []}

    # ✅ Use memory only to improve retrieval query (standalone rewrite)
    retrieval_question = (
        rewrite_question_with_context(question, chat_history)
        if use_conversation_memory
        else question
    )

    vs: Chroma = get_vectorstore()

    # Retrieve with scores (distance)
    docs_with_scores = vs.similarity_search_with_score(retrieval_question, k=k)

    # Convert distance to "relevance-ish"
    # (This is heuristic; good enough for toy project.)
    sources = _make_sources(docs_with_scores)

    if strict:
        best_dist = docs_with_scores[0][1] if docs_with_scores else 1.0
        best_rel = 1.0 - float(best_dist)
        if (not docs_with_scores) or (best_rel < min_relevance):
            return {
                "answer": "I can’t find this in the provided documents (based on retrieval). Try rephrasing or add more sources.",
                "sources": sources[:3],
            }

    docs = [d for d, _ in docs_with_scores]
    context = _format_context(docs)

    llm = ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key, temperature=0)

    user_prompt = f"""QUESTION:
{question}

CONTEXT:
{context}

Write the answer. Include citations like [1], [2] corresponding to the context blocks."""
    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {"answer": resp.content, "sources": sources}
