# rag/ingest.py
from __future__ import annotations

from typing import List, Optional
import os
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma


from rag.config import settings
from rag.loaders import (
    load_pdf_documents,
    load_text_file,
    load_url_document,
    load_url_list_file,
)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["chunk_hash"] = _content_hash(c.page_content)
    return chunks


def get_vectorstore() -> Chroma:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Put it in .env")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.persist_dir,
    )


def discover_files(data_dir: str) -> dict:
    """
    Discovers:
      - PDFs anywhere under data_dir
      - text files (.txt, .md) anywhere under data_dir
      - URLLinks.txt (expected at the root of data_dir; optional)

    NOTE: URLLinks.txt is treated specially and is excluded from "texts" by default
          (because it's just a list of URLs, usually not useful to embed).
    """
    base = Path(data_dir)
    if not base.exists() or not base.is_dir():
        raise ValueError(f"data_dir is not a directory: {data_dir}")

    pdfs = sorted(str(p) for p in base.rglob("*.pdf"))

    texts = []
    for ext in ("*.txt", "*.md"):
        texts.extend(str(p) for p in base.rglob(ext))
    texts = sorted(set(texts))

    url_file = str((base / "URLLinks.txt").resolve())
    # Exclude URLLinks.txt from texts so it doesn't pollute the index
    texts = [t for t in texts if Path(t).name != "URLLinks.txt"]

    return {"pdfs": pdfs, "texts": texts, "url_file": url_file}


def ingest(
    data_dir: Optional[str] = None,
    pdf_paths: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    reset_collection: bool = False,
) -> dict:
    """
    Build/update the index.

    Loads:
      - all PDFs/TXT/MD from data_dir (if provided)
      - URLLinks.txt from data_dir root (if present): one URL per line, supports # comments
      - plus optional additional pdf_paths / urls (if you want to pass extras)

    Then: chunk -> embed -> upsert into Chroma (persisted on disk).
    """
    pdf_paths = pdf_paths or []
    urls = urls or []

    discovered = {"pdfs": [], "texts": [], "url_file": None}
    if data_dir:
        discovered = discover_files(data_dir)

        # combine discovered PDFs with any provided pdf_paths
        pdf_paths = list(dict.fromkeys(discovered["pdfs"] + pdf_paths))  # dedup preserve order

        # auto-load URLs from URLLinks.txt if it exists
        auto_urls = load_url_list_file(discovered["url_file"]) if discovered.get("url_file") else []
        urls = list(dict.fromkeys(auto_urls + urls))  # combine + dedup

    vs = get_vectorstore()

    if reset_collection:
        # Toy reset: delete persistent dir contents (fast + simple).
        if os.path.isdir(settings.persist_dir):
            for root, _, files in os.walk(settings.persist_dir):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except OSError:
                        pass
        vs = get_vectorstore()

    raw_docs: List[Document] = []

    # PDFs (page-cited)
    for p in pdf_paths:
        raw_docs.extend(load_pdf_documents(p))

    # Text files (single doc each)
    if data_dir:
        for t in discovered["texts"]:
            raw_docs.extend(load_text_file(t))

    # URLs (scraped)
    for u in urls:
        raw_docs.extend(load_url_document(u))

    if not raw_docs:
        return {
            "raw_docs": 0,
            "chunks": 0,
            "added": 0,
            "pdfs": len(pdf_paths),
            "texts": len(discovered["texts"]) if data_dir else 0,
            "urls": len(urls),
            "persist_dir": settings.persist_dir,
            "data_dir": data_dir,
        }

    chunks = split_documents(raw_docs)

    # Dedup within this ingest run by chunk_hash
    seen = set()
    unique_chunks: List[Document] = []
    for c in chunks:
        h = c.metadata.get("chunk_hash")
        if h and h not in seen:
            seen.add(h)
            unique_chunks.append(c)

    # Use chunk_hash IDs to avoid duplicates across repeated runs
    ids = [c.metadata["chunk_hash"] for c in unique_chunks]

    vs.add_documents(unique_chunks, ids=ids)
    # No vs.persist() needed with langchain-chroma; persistence is handled via persist_directory.

    return {
        "raw_docs": len(raw_docs),
        "chunks": len(chunks),
        "added": len(unique_chunks),
        "pdfs": len(pdf_paths),
        "texts": len(discovered["texts"]) if data_dir else 0,
        "urls": len(urls),
        "persist_dir": settings.persist_dir,
        "data_dir": data_dir,
        "url_file": discovered.get("url_file"),
    }
