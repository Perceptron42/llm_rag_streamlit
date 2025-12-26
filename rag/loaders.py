from __future__ import annotations

from typing import List, Dict
import re
import requests
from bs4 import BeautifulSoup
import trafilatura
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """
    Returns Documents with metadata including:
      - source: filename/path
      - page: page number (0-index from loader)
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # normalize metadata
    for d in docs:
        d.metadata["source"] = d.metadata.get("source", pdf_path)
        # "page" already present in PyPDFLoader metadata
    return docs


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_url_list_file(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    urls = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    return urls


def load_url_document(url: str, timeout: int = 20) -> List[Document]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    html = r.text

    extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
    if extracted and len(extracted.strip()) > 200:
        text = extracted
    else:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    text = _clean_text(text)
    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": url,
                "type": "url",
            },
        )
    ]


def load_text_file(path: str) -> List[Document]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    text = _clean_text(text)
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": str(p),
                "type": "text",
            },
        )
    ]
