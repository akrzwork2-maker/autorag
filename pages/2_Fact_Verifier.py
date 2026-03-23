"""VerifAI — Fact Verifier"""

import streamlit as st
import plotly.graph_objects as go

from core.config import FACT_CHECK_EXAMPLES, verifier
from utils.styles import inject_css, top_bar, conf_badge

st.set_page_config(page_title="Fact Verifier", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")
inject_css()
top_bar("Fact Verifier", "DeBERTa NLI")


@st.cache_resource(show_spinner=False)
def _load():
    try:
        from core.embedder import Embedder
        from core.vectorstore import VectorStore
        from core.evaluator import Evaluator
        emb = Embedder()
        vs = VectorStore(embedder=emb)
        return vs, Evaluator(emb)
    except Exception as e:
        return str(e)


if "fv_run" not in st.session_state:
    st.session_state.fv_run = False

# Input
text_input = st.text_area("Paste text to verify", height=150, key="fv_input",
                           placeholder="Enter claims to fact-check against the knowledge base...")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    if st.button("Verify", type="primary", disabled=not text_input, use_container_width=True):
        st.session_state.fv_run = True
with c2:
    if st.button("Example: Accurate", use_container_width=True):
        st.session_state.fv_input = FACT_CHECK_EXAMPLES["accurate"]
        st.session_state.fv_run = True
        st.rerun()
with c3:
    if st.button("Example: Mixed", use_container_width=True):
        st.session_state.fv_input = FACT_CHECK_EXAMPLES["mixed"]
        st.session_state.fv_run = True
        st.rerun()
with c4:
    if st.button("Example: Inaccurate", use_container_width=True):
        st.session_state.fv_input = FACT_CHECK_EXAMPLES["inaccurate"]
        st.session_state.fv_run = True
        st.rerun()

if not st.session_state.fv_run or not text_input:
    st.session_state.fv_run = False
    st.stop()

comp = _load()
if isinstance(comp, str):
    st.error(comp)
    st.stop()
vs, evaluator = comp
st.session_state.fv_run = False

with st.spinner("Retrieving evidence..."):
    evidence = vs.query(text_input[:500], k=10)
if not evidence:
    st.warning("Knowledge Base is empty. Upload documents first.")
    st.stop()

with st.spinner("Running NLI verification..."):
    verdicts = evaluator.verify_text(text_input, evidence)
if not verdicts:
    st.info("No claims extracted.")
    st.stop()

if "fact_check_count" not in st.session_state:
    st.session_state.fact_check_count = 0
st.session_state.fact_check_count += 1

# Summary
verified = sum(1 for v in verdicts if v.verdict == "VERIFIED")
contradicted = sum(1 for v in verdicts if v.verdict == "CONTRADICTED")
unverifiable = sum(1 for v in verdicts if v.verdict == "UNVERIFIABLE")
total = len(verdicts)
avg_s = sum(v.nli_score for v in verdicts) / total

st.markdown("---")
st.markdown(f"""
<div class="metric-row">
    <div class="metric-item"><div class="label">Claims</div><div class="value">{total}</div></div>
    <div class="metric-item"><div class="label">Verified</div><div class="value" style="color:#2ecc71">{verified}</div></div>
    <div class="metric-item"><div class="label">Contradicted</div><div class="value" style="color:#e74c3c">{contradicted}</div></div>
    <div class="metric-item"><div class="label">Unverifiable</div><div class="value" style="color:#f39c12">{unverifiable}</div></div>
    <div class="metric-item"><div class="label">Avg NLI</div><div class="value">{avg_s:.3f}</div></div>
</div>
""", unsafe_allow_html=True)

# Charts
ch1, ch2 = st.columns(2)
with ch1:
    fig = go.Figure(go.Pie(
        labels=["Verified", "Contradicted", "Unverifiable"],
        values=[verified, contradicted, unverifiable],
        marker_colors=["#2ecc71", "#e74c3c", "#f39c12"],
        hole=0.45, textinfo="label+percent",
    ))
    fig.update_layout(height=280, template="plotly_dark", showlegend=False,
                      margin=dict(t=20, b=20, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with ch2:
    colors = ["#2ecc71" if v.verdict == "VERIFIED" else "#e74c3c" if v.verdict == "CONTRADICTED" else "#f39c12" for v in verdicts]
    fig2 = go.Figure(go.Bar(
        x=[f"C{i+1}" for i in range(total)], y=[v.nli_score for v in verdicts],
        marker_color=colors, text=[f"{v.nli_score:.2f}" for v in verdicts], textposition="outside",
    ))
    fig2.add_hline(y=verifier.verified_threshold, line_dash="dash", line_color="#2ecc71", opacity=0.4)
    fig2.add_hline(y=verifier.contradicted_threshold, line_dash="dash", line_color="#e74c3c", opacity=0.4)
    fig2.update_layout(yaxis_range=[0, 1.1], height=280, template="plotly_dark",
                       margin=dict(t=20, b=20, l=40, r=20),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# Claims
for i, v in enumerate(verdicts):
    dot_cls = v.verdict.lower()
    with st.expander(f"Claim {i+1}: {v.claim[:80]}{'...' if len(v.claim) > 80 else ''} — {v.verdict} ({v.nli_score:.3f})",
                     expanded=(v.verdict != "VERIFIED")):
        st.markdown(v.claim)
        st.progress(min(v.nli_score, 1.0))
        if v.evidence:
            st.caption(f"Evidence ({v.evidence_source}): {v.evidence}")
