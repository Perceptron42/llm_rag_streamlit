import streamlit as st
from rag.ingest import ingest, discover_files
from rag.qa import ask_question
from rag.config import settings
import os

st.set_page_config(page_title="Toy RAG (Directory)", layout="wide")
st.title("Toy RAG (Index from a Local Directory)")
st.sidebar.caption(f"Embedding model: {settings.embedding_model}")
st.sidebar.caption(f"Chat model: {settings.chat_model}")
key = os.getenv("OPENAI_API_KEY", "")
st.sidebar.caption(f"API key loaded: {'YES' if key else 'NO'} | endswith: {key[-6:] if key else 'N/A'}")


with st.sidebar:
    st.header("Index Settings")
    st.caption(f"Chroma dir: `{settings.persist_dir}`")

    data_dir = st.text_input(
        "Data directory (PDF/TXT/MD)",
        value="./data",
        help="Put your PDFs and text files here. Subfolders are included.",
    )

    reset = st.checkbox("Reset index on next build", value=False)

    st.divider()
    st.subheader("Optional URLs")
    url_text = st.text_area("One URL per line", height=120)

    st.divider()
    st.subheader("Retrieval")
    strict_mode = st.toggle("Strict mode (refuse if not found)", value=True)
    use_memory = st.toggle("Conversation memory (follow-up questions)", value=True)

    min_rel = st.slider("Min relevance (strict)", 0.0, 0.9, 0.35, 0.01)
    k = st.slider("Retrieved chunks (k)", 2, 12, 6, 1)

colA, colB = st.columns([1, 2], vertical_alignment="top")

with colA:
    st.subheader("Index")
    if st.button("Build / Refresh Index", type="primary"):
        urls = [u.strip() for u in (url_text or "").splitlines() if u.strip()]

        try:
            with st.spinner("Discovering files..."):
                found = discover_files(data_dir)
            st.write(f"Found: {len(found['pdfs'])} PDFs, {len(found['texts'])} text files")

            with st.spinner("Ingesting & embedding..."):
                stats = ingest(data_dir=data_dir, urls=urls, reset_collection=reset)

            st.success(
                f"Done. PDFs: {stats['pdfs']} | Text: {stats['texts']} | URLs: {stats['urls']} "
                f"| Raw docs: {stats['raw_docs']} | Chunks: {stats['chunks']} | Added: {stats['added']}"
            )
        except Exception as e:
            st.error(f"Index build failed: {e}")

    st.info("Tip: create a folder named `data/` next to this script and drop files there.")

with colB:
    st.subheader("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask a question about your local docs...")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(
                    q,
                    strict=strict_mode,
                    min_relevance=min_rel,
                    k=k,
                    use_conversation_memory=use_memory,
                    chat_history=st.session_state.messages,
                )

            st.markdown(result["answer"])

            if result.get("sources"):
                with st.expander("Sources"):
                    for i, s in enumerate(result["sources"], start=1):
                        page_str = f" (page {s.page})" if s.page else ""
                        st.markdown(f"**{i}. {s.source}{page_str}**  \nscore: `{s.score:.4f}`")
                        st.caption(s.snippet)

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
