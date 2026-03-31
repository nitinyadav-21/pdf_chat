import streamlit as st
import os
import time
import tempfile
from pathlib import Path

from pdf_processor import extract_text_from_pdfs
from vector_store import VectorStore
from llm_handler import get_answer

@st.cache_resource(show_spinner=False)
def get_cached_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(VectorStore.MODEL_NAME)

st.set_page_config(
    page_title="PDF Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d2f3e;
    }

    /* Chat message bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .assistant-bubble {
        background-color: #1e2132;
        border: 1px solid #2d2f3e;
        color: #e0e0e0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        word-wrap: break-word;
    }
    .source-tag {
        display: inline-block;
        background-color: #2d2f3e;
        color: #9b9fc4;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        margin: 4px 2px 0 2px;
    }
    .chat-container {
        padding: 10px 0;
    }
    /* Upload area */
    .upload-hint {
        text-align: center;
        color: #6b6f8a;
        font-size: 13px;
        margin-top: 8px;
    }
    /* Status badge */
    .status-ready {
        background-color: #1a3a2a;
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        border: 1px solid #4ade80;
    }
    .status-pending {
        background-color: #2a1a1a;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        border: 1px solid #f87171;
    }
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "processing" not in st.session_state:
    st.session_state.processing = False


with st.sidebar:
    st.markdown("## PDF Chat")
    st.markdown("*Ask questions across your documents*")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.markdown(f"• `{f.name}` ({size_kb:.1f} KB)")

    st.divider()

    process_btn = st.button(
        "Process PDFs",
        use_container_width=True,
        type="primary",
        disabled=(not uploaded_files),
    )

    if st.session_state.processed_files:
        st.markdown(
            '<span class="status-ready">✓ Ready to chat</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<small style='color:#6b6f8a;'>Indexed: {', '.join(st.session_state.processed_files)}</small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pending">○ No PDFs indexed</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.expander("Settings", expanded=False):
        chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, 50)
        top_k = st.slider("Top chunks to retrieve", 1, 8, 4)
        st.caption("Larger chunks = more context per result. Higher k = more sources checked.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        "<small style='color:#4b4f6a;'>Powered by HuggingFace + FAISS<br>"
        "Deployed on Render • 100% free</small>",
        unsafe_allow_html=True,
    )


if process_btn and uploaded_files:
    with st.spinner("Reading and indexing PDFs...."):
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "pdf_chat_uploads"
            tmp_dir.mkdir(exist_ok=True)

            file_paths = []
            for f in uploaded_files:
                dest = tmp_dir / f.name
                dest.write_bytes(f.getvalue())
                file_paths.append(str(dest))

            chunks, metadata = extract_text_from_pdfs(file_paths, chunk_size=chunk_size)

            if not chunks:
                st.error("No text could be extracted from the uploaded PDFs.")
            else:
                vs = VectorStore()
                vs.build(chunks, metadata)
                st.session_state.vector_store = vs
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.messages = []  
                st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s)!")
                time.sleep(1)
                st.rerun()

        except Exception as e:
            st.error(f"Error processing PDFs: {e}")

st.markdown("## Chat with your PDFs")

if not st.session_state.vector_store:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px; color: #6b6f8a;'>
            <div style='font-size: 64px; margin-bottom: 16px;'>📄</div>
            <h3 style='color: #9b9fc4;'>No PDFs loaded yet</h3>
            <p>Upload your PDFs in the sidebar and click <strong style='color:#667eea;'>Process PDFs</strong> to get started.</p>
            <br>
            <p style='font-size:13px;'>You can upload multiple PDFs and ask questions across all of them at once.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    chat_area = st.container()
    with chat_area:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-container"><div class="user-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                sources_html = ""
                if msg.get("sources"):
                    tags = "".join(
                        f'<span class="source-tag">{s}</span>'
                        for s in msg["sources"]
                    )
                    sources_html = f"<div style='margin-top:8px;'>{tags}</div>"
                st.markdown(
                    f'<div class="chat-container">'
                    f'<div class="assistant-bubble">{msg["content"]}{sources_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Ask a question…",
                label_visibility="collapsed",
                placeholder="e.g. What are the main conclusions in the report?",
            )
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True, type="primary")

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

        with st.spinner("🔍 Searching documents…"):
            try:
                vs: VectorStore = st.session_state.vector_store
                results = vs.search(user_input.strip(), k=top_k)

                context_chunks = [r["text"] for r in results]
                sources = list(dict.fromkeys(r["source"] for r in results))  

                answer = get_answer(user_input.strip(), context_chunks)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Error generating answer: {e}",
                        "sources": [],
                    }
                )

        st.rerun()
