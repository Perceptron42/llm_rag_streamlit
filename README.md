# 🧠 RAG Document Q&A Application

A Retrieval-Augmented Generation (RAG) system built with Streamlit that enables you to ask questions about your local documents (PDFs, text files, markdown) and web URLs using OpenAI's language models.

You can put all the data about your project in the `data` directory. The data directory is a subdirectory of the project root directory. 

For URLs, you can add them in the sidebar. One URL per line. You can also add them in the `URLLinks.txt` file in the `data` directory. 

**Models** are configurable in the `.env` file. The default models are `gpt-5-nano` for chat and `text-embedding-3-small` for embeddings. We are using `gpt-5-nano` for chat and `text-embedding-3-small` for embeddings and they have been picked for their cost-effectiveness.
For example `gpt-5-nano` is $0.05 for 1M tokens (input) and $0.40 for 1M tokens (output)  and `text-embedding-3-small` is $0.02 for 1M tokens. 

**Vector database** is configurable in the `.env` file. The default vector database is `chromadb`. ChromaDB is the "memory" that lets AI answer questions about your specific documents accurately, even when users ask in different ways. It's the bridge between your content and intelligent, conversational AI responses. In our app: Users drop documents in a folder → ChromaDB indexes them → Users can ask questions in plain English → AI finds relevant info and answers accurately with source citations. 🎯

Why ChromaDB Specifically?
- Open source - No vendor lock-in
- Easy to use - Simple Python API
- Runs locally - No need for cloud infrastructure (though it can scale to cloud)
- Built for AI - Designed specifically for RAG (Retrieval-Augmented Generation) applications

**Conversation memory** is configurable in the `.env` file. The default conversation memory is `true`. When enabled, the system rewrites follow-up questions into standalone queries using recent chat history (last 6 messages). For example, if you ask "What is Python?" followed by "What about its history?", the second question is automatically rewritten to "What is the history of Python?" before retrieval, ensuring accurate context-aware results without storing the entire conversation in the vector database.

**Strict mode** is configurable in the `.env` file. The default strict mode is `true`. When enabled, the system refuses to answer if the relevance is too low (prevents hallucinations).

## ✨ Features

- 📄 **Multi-format document support**: PDF, TXT, and MD files
- 🌐 **Web scraping**: Index content from URLs
- 💬 **Conversational memory**: Follow-up questions with context awareness
- 🎯 **Strict mode**: Refuses to answer when relevance is too low (prevents hallucinations)
- 📊 **Source citations**: Shows which documents were used with relevance scores
- 💾 **Persistent vector database**: Index survives app restarts with ChromaDB
- 🔄 **Deduplication**: Hash-based chunk deduplication across re-indexing

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd llmrag_experimients
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_DIR=./chroma_db
CHROMA_COLLECTION=rag_toy
CHUNK_SIZE=900
CHUNK_OVERLAP=150
```

5. **Create data directory**
```bash
mkdir data
```

### Usage

1. **Start the application**
```bash
streamlit run app_streamlit.py
```

2. **Index your documents**
   - Place PDFs, TXT, or MD files in the `./data` directory
   - (Optional) Add URLs in the sidebar
   - Click "Build / Refresh Index"

3. **Ask questions**
   - Type your question in the chat interface
   - View answers with source citations
   - Enable conversation memory for follow-up questions

## 📁 Project Structure

```
llmrag_experimients/
├── app_streamlit.py          # Main Streamlit application
├── rag/
│   ├── __init__.py
│   ├── config.py             # Configuration & settings
│   ├── ingest.py             # Document indexing pipeline
│   ├── loaders.py            # PDF/text/URL loaders
│   └── qa.py                 # Question answering logic
├── data/                     # Your documents (PDFs, TXT, MD)
├── chroma_db/                # Persistent vector database
├── .env                      # Environment variables (not in git)
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Sidebar Settings

- **Data directory**: Path to your documents (default: `./data`)
- **Reset index**: Clear existing index before rebuilding
- **Optional URLs**: Add web pages to index (one per line)
- **Strict mode**: Refuse answers if relevance < threshold
- **Conversation memory**: Enable follow-up questions with context
- **Min relevance**: Threshold for strict mode (0.0 - 0.9)
- **Retrieved chunks (k)**: Number of document chunks to retrieve (2-12)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for answers |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_DIR` | `./chroma_db` | Vector database directory |
| `CHROMA_COLLECTION` | `rag_toy` | Collection name |
| `CHUNK_SIZE` | `900` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |

## 🛠️ Technologies

- **[Streamlit](https://streamlit.io/)**: Web UI framework
- **[LangChain](https://python.langchain.com/)**: RAG framework
- **[ChromaDB](https://www.trychroma.com/)**: Vector database
- **[OpenAI](https://openai.com/)**: Embeddings & chat models
- **[PyPDF](https://pypdf.readthedocs.io/)**: PDF parsing
- **[Trafilatura](https://trafilatura.readthedocs.io/)**: Web scraping

## 📝 How It Works

### Architecture Diagram

```mermaid
graph TB
    subgraph "User Interface (Streamlit)"
        UI[User Interface]
        SIDEBAR[Sidebar Controls]
        CHAT[Chat Interface]
    end

    subgraph "Indexing Pipeline"
        DATA[("Data Sources<br/>PDFs, TXT, MD, URLs")]
        DISCOVER["discover_files()<br/>Find all documents"]
        LOAD["Loaders<br/>load_pdf_documents()<br/>load_text_file()<br/>load_url_document()"]
        SPLIT["split_documents()<br/>Chunk into ~900 chars<br/>with 150 overlap"]
        HASH["Add chunk_hash<br/>SHA-256 for dedup"]
        EMBED["OpenAI Embeddings<br/>text-embedding-3-small"]
        CHROMA[("ChromaDB<br/>Vector Database<br/>./chroma_db")]
    end

    subgraph "Question Answering Pipeline"
        QUESTION["User Question"]
        MEMORY{"Conversation<br/>Memory?"}
        REWRITE["rewrite_question()<br/>Make standalone"]
        SEARCH["Vector Search<br/>similarity_search_with_score()"]
        STRICT{"Strict Mode?"}
        RELEVANCE{"Relevance ><br/>Threshold?"}
        FORMAT["_format_context()<br/>Create numbered citations"]
        LLM["OpenAI Chat<br/>gpt-4o-mini"]
        ANSWER["Answer + Sources"]
    end

    %% Indexing Flow
    UI --> |Build Index| DATA
    DATA --> DISCOVER
    DISCOVER --> LOAD
    LOAD --> SPLIT
    SPLIT --> HASH
    HASH --> EMBED
    EMBED --> CHROMA

    %% Question Flow
    CHAT --> QUESTION
    QUESTION --> MEMORY
    MEMORY --> |Yes| REWRITE
    MEMORY --> |No| SEARCH
    REWRITE --> SEARCH
    SEARCH --> |Top-k chunks| STRICT
    STRICT --> |Yes| RELEVANCE
    STRICT --> |No| FORMAT
    RELEVANCE --> |Pass| FORMAT
    RELEVANCE --> |Fail| ANSWER
    FORMAT --> LLM
    LLM --> ANSWER
    ANSWER --> CHAT

    %% Database connections
    CHROMA -.-> |Read vectors| SEARCH

    style CHROMA fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style LLM fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style EMBED fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style ANSWER fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

### Indexing Pipeline

1. **Discovery**: Recursively finds PDFs, TXT, and MD files in `data/`
2. **Loading**: Extracts text from documents and web pages
3. **Chunking**: Splits documents into 900-character chunks with 150-char overlap
4. **Embedding**: Converts chunks to vector embeddings via OpenAI
5. **Storage**: Stores in ChromaDB with hash-based deduplication

### Question Answering

1. **Optional rewrite**: Converts follow-up questions to standalone questions using chat history
2. **Retrieval**: Searches vector database for top-k similar chunks
3. **Relevance check**: If strict mode enabled, refuses if best match < threshold
4. **Generation**: LLM generates answer using retrieved context
5. **Citations**: Returns answer with source references

## 🔒 Security

- `.env` file is **gitignored** (never commit API keys!)
- Virtual environment excluded from version control
- User data in `data/` directory not tracked

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

**Happy querying!** 🎉
