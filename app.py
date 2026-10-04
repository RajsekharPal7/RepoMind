"""
RepoMind - Streamlit UI
Sidebar: index a repo (local path or GitHub URL).
Main: chat with the indexed codebase, with sources shown per answer.
"""
import streamlit as st

from config import settings
from src.rag import pipeline

st.set_page_config(page_title="RepoMind", page_icon="🧠", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str, "sources": [...]}]
if "active_repo" not in st.session_state:
    st.session_state.active_repo = None

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("🧠 RepoMind")
    st.caption("GitHub / codebase RAG assistant")

    st.subheader("1. Index a repository")
    source_input = st.text_input(
        "Local path or GitHub URL",
        placeholder="https://github.com/user/repo  or  /path/to/local/repo",
    )
    if st.button("Index repo", use_container_width=True, type="primary"):
        if not source_input.strip():
            st.warning("Enter a path or URL first.")
        elif not settings.openai_api_key:
            st.error("OPENAI_API_KEY is not set. Add it to your .env file.")
        else:
            with st.spinner("Loading, chunking and embedding files..."):
                try:
                    result = pipeline.index_source(source_input.strip())
                    st.session_state.active_repo = result["repo_name"]
                    st.success(
                        f"Indexed '{result['repo_name']}': "
                        f"{result['files_loaded']} files -> {result['chunks_indexed']} chunks"
                    )
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.subheader("2. Choose active repo")
    indexed = pipeline.list_indexed_repos()
    if indexed:
        default_idx = indexed.index(st.session_state.active_repo) if st.session_state.active_repo in indexed else 0
        st.session_state.active_repo = st.selectbox("Indexed repos", indexed, index=default_idx)
    else:
        st.info("No repos indexed yet.")

    top_k = st.slider("Chunks to retrieve (top_k)", 2, 15, settings.top_k)

    st.divider()
    st.caption(f"Chat model: `{settings.chat_model}`")
    st.caption(f"Embed model: `{settings.embed_model}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ main --
st.header("Chat with your codebase")

if not st.session_state.active_repo:
    st.info("Index a repository in the sidebar to get started.")
else:
    st.caption(f"Active repo: **{st.session_state.active_repo}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s['file_path']}` (lines {s['start_line']}-{s['end_line']})")

question = st.chat_input("Ask something about the indexed codebase...")
if question:
    if not st.session_state.active_repo:
        st.warning("Index and select a repo first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    result = pipeline.ask(
                        st.session_state.active_repo, question, top_k=top_k, history=history
                    )
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("Sources"):
                            for s in result["sources"]:
                                st.markdown(f"- `{s['file_path']}` (lines {s['start_line']}-{s['end_line']})")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
