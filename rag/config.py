from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Where Chroma persists
    persist_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "rag_toy")

    # Chunking defaults
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))


settings = Settings()
