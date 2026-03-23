"""VerifAI — Dashboard"""

import streamlit as st
from utils.styles import inject_css, top_bar

st.set_page_config(page_title="VerifAI", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")
inject_css()
top_bar("VerifAI", "AutoRAG++")

if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "fact_check_count" not in st.session_state:
    st.session_state.fact_check_count = 0


@st.cache_resource(show_spinner=False)
def _kb_stats():
    try:
        from core.embedder import Embedder
        from core.vectorstore import VectorStore
        return VectorStore(embedder=Embedder()).stats()
    except Exception:
        return {"total_chunks": 0, "total_documents": 0}


stats = _kb_stats()
history = st.session_state.query_history
avg_cf = sum(q["confidence"] for q in history) / len(history) if history else 0

# Metrics
st.markdown(f"""
<div class="metric-row">
    <div class="metric-item"><div class="label">Queries</div><div class="value">{len(history)}</div></div>
    <div class="metric-item"><div class="label">Avg Confidence</div><div class="value">{avg_cf:.2f}</div></div>
    <div class="metric-item"><div class="label">KB Chunks</div><div class="value">{stats.get("total_chunks", 0)}</div></div>
    <div class="metric-item"><div class="label">Documents</div><div class="value">{stats.get("total_documents", 0)}</div></div>
    <div class="metric-item"><div class="label">Fact Checks</div><div class="value">{st.session_state.fact_check_count}</div></div>
</div>
""", unsafe_allow_html=True)

if history:
    import plotly.graph_objects as go
    last = history[-10:]
    fig = go.Figure(go.Bar(
        x=[f"Q{i+1}" for i in range(len(last))],
        y=[q["confidence"] for q in last],
        marker_color=["#1a73e8" if q["confidence"] >= 0.7 else "#f39c12" if q["confidence"] >= 0.5 else "#e74c3c" for q in last],
        text=[f"{q['confidence']:.2f}" for q in last], textposition="outside",
    ))
    fig.add_hline(y=0.70, line_dash="dash", line_color="#555", annotation_text="τ=0.70")
    fig.update_layout(yaxis_range=[0, 1.1], height=250, template="plotly_dark",
                      margin=dict(t=30, b=20, l=40, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**Q&A Engine**")
    st.caption("Pipeline with live tracing")
    if st.button("Open", key="nav_qa", use_container_width=True, type="primary"):
        st.switch_page("pages/1_AutoRAG_QA.py")
with c2:
    st.markdown("**Fact Verifier**")
    st.caption("NLI claim verification")
    if st.button("Open", key="nav_fv", use_container_width=True):
        st.switch_page("pages/2_Fact_Verifier.py")
with c3:
    st.markdown("**Pipeline X-Ray**")
    st.caption("3-way comparison")
    if st.button("Open", key="nav_xr", use_container_width=True):
        st.switch_page("pages/3_Pipeline_XRay.py")
with c4:
    st.markdown("**Knowledge Base**")
    st.caption("Upload & manage docs")
    if st.button("Open", key="nav_kb", use_container_width=True):
        st.switch_page("pages/4_Knowledge_Base.py")
