"""VerifAI — Knowledge Base"""

import streamlit as st
import pandas as pd

from utils.styles import inject_css, top_bar

st.set_page_config(page_title="Knowledge Base", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")
inject_css()
top_bar("Knowledge Base", "ChromaDB")


@st.cache_resource(show_spinner=False)
def _load():
    from core.embedder import Embedder
    from core.vectorstore import VectorStore
    emb = Embedder()
    return VectorStore(embedder=emb)


vs = _load()
stats = vs.stats()

# Stats row
st.markdown(f"""
<div class="metric-row">
    <div class="metric-item"><div class="label">Chunks</div><div class="value">{stats['total_chunks']}</div></div>
    <div class="metric-item"><div class="label">Documents</div><div class="value">{stats['total_documents']}</div></div>
    <div class="metric-item"><div class="label">Seed</div><div class="value">{stats['seed_docs']}</div></div>
    <div class="metric-item"><div class="label">Auto-fetched</div><div class="value">{stats['auto_fetched_docs']}</div></div>
    <div class="metric-item"><div class="label">Uploaded</div><div class="value">{stats['uploaded_docs']}</div></div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_docs, tab_upload, tab_search, tab_seed = st.tabs(["Documents", "Upload PDF", "Search", "Re-Seed"])

with tab_docs:
    docs = vs.list_documents()
    if docs:
        df = pd.DataFrame([{"Document": d["doc_name"], "Source": d["source"], "Chunks": d["chunk_count"]} for d in docs])
        st.dataframe(df, use_container_width=True, hide_index=True)

        dc1, dc2 = st.columns([3, 1])
        with dc1:
            target = st.selectbox("Select document to remove", [d["doc_name"] for d in docs], label_visibility="collapsed")
        with dc2:
            if st.button("Delete", type="secondary", use_container_width=True):
                vs.delete_document(target)
                st.rerun()
    else:
        st.info("Knowledge Base is empty.")

with tab_upload:
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded:
        doc_name = uploaded.name.rsplit(".", 1)[0]
        if vs.has_document(doc_name):
            st.warning(f"{doc_name} already exists. Delete it first.")
        elif st.button("Index", type="primary"):
            with st.spinner("Processing..."):
                from utils.pdf_loader import load_pdf_bytes
                chunks = load_pdf_bytes(uploaded.read(), doc_name=doc_name, source="uploaded")
                if chunks:
                    vs.add_chunks(chunks)
                    st.success(f"Indexed {len(chunks)} chunks from {doc_name}")
                    st.rerun()
                else:
                    st.error("No text extracted from PDF.")

with tab_search:
    sq = st.text_input("Search query", placeholder="e.g., transformer self-attention mechanism")
    sk = st.slider("Results (k)", 1, 20, 5)
    if st.button("Search", disabled=not sq):
        results = vs.query(sq, k=sk)
        if results:
            for i, r in enumerate(results):
                with st.expander(f"#{i+1} {r['metadata'].get('doc_name', '?')} — {r['similarity']:.4f}"):
                    st.markdown(r["text"])
        else:
            st.info("No results.")

with tab_seed:
    st.caption("Re-run Wikipedia seeding to add base articles.")
    if st.button("Re-Seed Knowledge Base", type="primary"):
        with st.spinner("Seeding..."):
            from utils.wiki import WikiFetcher
            n = WikiFetcher().seed_knowledge_base(vs)
            st.success(f"Seeded {n} new chunks")
            st.rerun()
