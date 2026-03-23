"""VerifAI — Pipeline X-Ray"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.config import XRAY_QUERIES, calibration
from utils.styles import inject_css, top_bar, conf_badge, render_pipeline_step

st.set_page_config(page_title="Pipeline X-Ray", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")
inject_css()
top_bar("Pipeline X-Ray", "3-Way Comparison")


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

if "xray_run" not in st.session_state:
    st.session_state.xray_run = False

query = st.text_input("Enter a query to compare across pipeline modes", key="xray_input",
                       placeholder="e.g., How does retrieval-augmented generation work?")

cols = st.columns([1, 1, 1] + [1] * min(len(XRAY_QUERIES), 3))
with cols[0]:
    if st.button("Compare", type="primary", disabled=not query, use_container_width=True):
        st.session_state.xray_run = True
for i, q in enumerate(XRAY_QUERIES[:3]):
    with cols[i + 1]:
        if st.button(q[:35] + "...", key=f"xray_{i}", use_container_width=True):
            st.session_state.xray_input = q
            st.session_state.xray_run = True
            st.rerun()

if not st.session_state.xray_run or not query:
    st.session_state.xray_run = False
    st.stop()

pipeline = _load_pipeline()
if isinstance(pipeline, str):
    st.error(pipeline); st.stop()
st.session_state.xray_run = False

progress = st.progress(0, text="AutoRAG++...")
result_full = pipeline.run(query)
progress.progress(33, text="Basic RAG...")
result_basic = pipeline.run_basic_rag(query)
progress.progress(66, text="LLM-only...")
result_llm = pipeline.run_llm_only(query)
progress.progress(100, text="Done")

results = {"AutoRAG++": result_full, "Basic RAG": result_basic, "LLM-Only": result_llm}
modes = list(results.keys())

st.markdown("---")

# Charts
ch1, ch2 = st.columns(2)
with ch1:
    fig = go.Figure()
    for name, key, color in [("S_c", "semantic_consistency", "#3498db"),
                              ("F_c", "factual_correctness", "#9b59b6"),
                              ("C_f", "confidence", "#2ecc71")]:
        fig.add_trace(go.Bar(name=name, x=modes,
                             y=[getattr(results[m], key) for m in modes],
                             text=[f"{getattr(results[m], key):.3f}" for m in modes],
                             textposition="outside", marker_color=color))
    fig.add_hline(y=calibration.confidence_threshold, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_layout(yaxis_range=[0, 1.15], barmode="group", height=320, template="plotly_dark",
                      margin=dict(t=20, b=30, l=40, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

with ch2:
    fig2 = go.Figure(go.Bar(x=modes, y=[results[m].total_time_ms for m in modes],
                            text=[f"{results[m].total_time_ms:.0f}ms" for m in modes],
                            textposition="outside", marker_color="#e74c3c"))
    fig2.update_layout(height=320, template="plotly_dark",
                       margin=dict(t=20, b=30, l=40, r=20),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# Side-by-side answers
c1, c2, c3 = st.columns(3)
for col, (mode, res) in zip([c1, c2, c3], results.items()):
    with col:
        st.markdown(f"**{mode}**")
        st.markdown(f"""
<div class="metric-row">
    <div class="metric-item"><div class="label">C_f</div><div class="value">{res.confidence:.3f}</div></div>
    <div class="metric-item"><div class="label">Sources</div><div class="value">{res.total_sources_used}</div></div>
    <div class="metric-item"><div class="label">Time</div><div class="value">{res.total_time_ms:.0f}ms</div></div>
</div>""", unsafe_allow_html=True)
        if res.error:
            st.error(res.error)
        elif res.answer:
            st.markdown(f'<div class="answer-card">{res.answer}</div>', unsafe_allow_html=True)
        if res.sources:
            with st.expander(f"Sources ({len(res.sources)})"):
                for s in res.sources:
                    st.caption(f"[{s['rank']}] {s['doc_name']} ({s['hybrid_score']:.3f})")

# Comparison table
st.markdown("---")
df = pd.DataFrame({
    "Metric": ["S_c", "F_c", "C_f", "Sources", "Cycles", "Time (ms)", "Model"],
    **{m: [f"{getattr(r, 'semantic_consistency', 0):.3f}",
           f"{getattr(r, 'factual_correctness', 0):.3f}",
           f"{getattr(r, 'confidence', 0):.3f}",
           str(r.total_sources_used), str(r.calibration_cycles),
           f"{r.total_time_ms:.0f}", r.model_used]
       for m, r in results.items()}
})
st.dataframe(df, use_container_width=True, hide_index=True)

# Trace
if result_full.trace:
    with st.expander("AutoRAG++ Pipeline Trace"):
        for step in result_full.trace:
            st.markdown(render_pipeline_step(step, step == result_full.trace[-1]), unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### 🔬 Pipeline X-Ray")
    st.caption("3-Way Comparison")
    st.divider()
    st.markdown("**Modes:**")
    st.markdown(
        "- **AutoRAG++**: Full pipeline with\n"
        "  calibration loop (Algorithm 1)\n"
        "- **Basic RAG**: Single retrieval,\n"
        "  cosine-only, no calibration\n"
        "- **LLM-Only**: No retrieval at all,\n"
        "  pure language model"
    )
    st.divider()
    st.markdown("**Key Insight:**")
    st.markdown(
        "AutoRAG++ should show higher C_f scores "
        "because the calibration loop adjusts k and λ "
        "when confidence is below τ, fetching new "
        "documents until the answer is well-grounded."
    )
