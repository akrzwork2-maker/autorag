"""VerifAI — Q&A Engine with live pipeline visualization."""

import streamlit as st
from utils.styles import inject_css, top_bar, render_pipeline_step, conf_badge

st.set_page_config(page_title="Q&A Engine", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")
inject_css()
top_bar("Q&A Engine", "AutoRAG++")


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    try:
        from core.embedder import Embedder
        from core.vectorstore import VectorStore
        from core.retriever import Retriever
        from core.reasoner import Reasoner
        from core.evaluator import Evaluator
        from core.calibrator import Calibrator
        from core.pipeline import Pipeline
        from utils.wiki import WikiFetcher

        emb = Embedder()
        vs = VectorStore(embedder=emb)
        return Pipeline(emb, vs, Retriever(vs, emb), Reasoner(), Evaluator(emb), Calibrator(), WikiFetcher())
    except Exception as e:
        return str(e)


# Session state
if "qa_run" not in st.session_state:
    st.session_state.qa_run = False

# Input
query = st.text_input("Ask a question", placeholder="e.g., How does self-attention work in transformers?", key="qa_input", label_visibility="collapsed")

if st.button("Run Pipeline", type="primary", disabled=not query, use_container_width=False):
    st.session_state.qa_run = True

if not st.session_state.qa_run or not query:
    st.session_state.qa_run = False
    st.stop()

# ── Load pipeline ──
pipeline = _load_pipeline()
if isinstance(pipeline, str):
    st.error(f"Model loading failed: {pipeline}")
    st.stop()

st.session_state.qa_run = False

# ── Live pipeline execution ──
col_pipeline, col_result = st.columns([2, 3])

with col_pipeline:
    st.markdown("##### Pipeline Trace")
    timeline_container = st.container()
    timeline_html = []

    def on_step(step):
        is_last = step.step_type.value == "done"
        timeline_html.append(render_pipeline_step(step, is_last=is_last))
        with timeline_container:
            st.markdown("".join(timeline_html), unsafe_allow_html=True)

result = pipeline.run(query, on_step=on_step)

# Save history
if "query_history" not in st.session_state:
    st.session_state.query_history = []
st.session_state.query_history.append({
    "query": result.query,
    "confidence": result.confidence,
    "model": result.model_used,
    "cycles": result.calibration_cycles,
    "sources": result.total_sources_used,
})

with col_result:
    if result.error:
        st.error(result.error)
        st.stop()

    # Confidence + metrics
    st.markdown(f"##### Answer {conf_badge(result.confidence)}", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-item"><div class="label">S_c</div><div class="value">{result.semantic_consistency:.3f}</div></div>
        <div class="metric-item"><div class="label">F_c</div><div class="value">{result.factual_correctness:.3f}</div></div>
        <div class="metric-item"><div class="label">C_f</div><div class="value">{result.confidence:.3f}</div></div>
        <div class="metric-item"><div class="label">Time</div><div class="value">{result.total_time_ms:.0f}ms</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="answer-card">{result.answer}</div>', unsafe_allow_html=True)

    # Model info
    st.caption(f"{result.model_used} · {result.total_sources_used} sources · {result.calibration_cycles} calibration cycles")

# Sources
if result.sources:
    with st.expander(f"Sources ({len(result.sources)})"):
        for src in result.sources:
            st.markdown(f"""
            <div class="source-chip">
                <span class="name">[{src['rank']}] {src['doc_name']}</span>
                <span class="score"> · {src['hybrid_score']:.4f}</span>
                <div style="color:#888; font-size:0.78rem; margin-top:4px;">{src['text'][:200]}...</div>
            </div>
            """, unsafe_allow_html=True)
