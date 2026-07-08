import shutil
from pathlib import Path

import streamlit as st

from config import SAMPLE_DOCS_DIR
from src.pipeline import InsightRAG

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="InsightRAG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 InsightRAG")
st.caption("Hybrid RAG Research Assistant")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = None

if "ready" not in st.session_state:
    st.session_state.ready = False

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True,
    ):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            st.stop()

        # Remove old PDFs
        if SAMPLE_DOCS_DIR.exists():
            shutil.rmtree(SAMPLE_DOCS_DIR)

        SAMPLE_DOCS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save uploaded PDFs
        for file in uploaded_files:
            save_path = SAMPLE_DOCS_DIR / file.name

            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

        with st.spinner("Building knowledge base..."):

            rag = InsightRAG(SAMPLE_DOCS_DIR)
            rag.build_index()

            st.session_state.rag = rag
            st.session_state.ready = True

        st.success("Knowledge Base Ready!")

# ---------------------------------------------------
# Main Area
# ---------------------------------------------------

if not st.session_state.ready:

    st.info(
        "👈 Upload PDF documents and click **Build Knowledge Base**."
    )

    st.stop()

question = st.text_input(
    "💬 Ask a Question",
    placeholder="Example: What is polymorphism?",
)

if st.button(
    "Ask",
    use_container_width=True,
):

    if not question.strip():

        st.warning("Please enter a question.")

        st.stop()

    with st.spinner("Generating answer..."):

        response = st.session_state.rag.ask(question)

    tab1, tab2, tab3 = st.tabs(
        [
            "📝 Answer",
            "📚 Sources",
            "📊 Insights",
        ]
    )

    # ---------------------------------------------------
    # Answer
    # ---------------------------------------------------

    with tab1:

        st.markdown(response.answer)

    # ---------------------------------------------------
    # Sources
    # ---------------------------------------------------

    with tab2:

        if response.citations:

            for citation in response.citations:

                st.markdown(
                    f"📄 **{citation.filename}**  \n"
                    f"Page **{citation.page}**"
                )

        else:

            st.info("No sources available.")

    # ---------------------------------------------------
    # Insights
    # ---------------------------------------------------

    with tab3:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Confidence",
                response.confidence,
            )

        with col2:
            st.metric(
                "Retrieved",
                response.retrieval.retrieved_documents,
            )

        with col3:
            st.metric(
                "Reranked",
                response.retrieval.reranked_documents,
            )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")
st.caption(
    "InsightRAG • Hybrid RAG (FAISS + BM25) • Cross-Encoder Reranking • Groq Llama 3.3"
)