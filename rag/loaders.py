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
    """
    Cleans extracted text by normalizing whitespace.
    
    Removes excessive spaces, tabs, and newlines to create
    cleaner, more readable document chunks.
    
    Args:
        text: Raw text to clean
    
    Returns:
        str: Cleaned text with normalized whitespace
    
    Transformations:
        - Multiple spaces/tabs → single space
        - 3+ consecutive newlines → double newline
        - Leading/trailing whitespace removed
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_url_list_file(path: str) -> list[str]:
    """
    Loads URLs from a text file (typically URLLinks.txt).
    
    Each line should contain one URL. Lines starting with '#' are
    treated as comments and ignored. Empty lines are also skipped.
    
    Args:
        path: Absolute path to the URL list file
    
    Returns:
        list[str]: List of URLs found in the file (empty if file doesn't exist)
    
    Example file format:
        # Main documentation
        https://example.com/docs
        https://example.com/api
        
        # Blog posts
        https://blog.example.com/post1
    """
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
    """
    Scrapes and loads content from a web URL.
    
    Uses trafilatura for intelligent content extraction (preferred method),
    falling back to BeautifulSoup if trafilatura doesn't extract enough text.
    Removes script, style, and noscript tags before extraction.
    
    Args:
        url: Web page URL to scrape
        timeout: Request timeout in seconds (default: 20)
    
    Returns:
        List[Document]: Single-element list with scraped content,
                       or empty list if no text extracted
    
    Raises:
        requests.exceptions.HTTPError: If URL returns error status
        requests.exceptions.Timeout: If request exceeds timeout
    
    Metadata:
        - source: The URL
        - type: "url"
    """
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
    """
    Loads content from a text file (.txt, .md, etc.).
    
    Reads the entire file as a single document with UTF-8 encoding.
    Invalid characters are ignored gracefully.
    
    Args:
        path: Absolute path to the text file
    
    Returns:
        List[Document]: Single-element list with file content,
                       or empty list if file is empty after cleaning
    
    Metadata:
        - source: Absolute file path
        - type: "text"
    
    Supported formats:
        - .txt (plain text)
        - .md (markdown)
        - Any UTF-8 text file
    """
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
