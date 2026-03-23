"""Shared CSS and styling for VerifAI UI."""

GLOBAL_CSS = """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
}

/* Top bar */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid #222;
    margin-bottom: 1.2rem;
}
.top-bar h1 {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0;
    color: #f0f0f0;
    letter-spacing: -0.5px;
}
.top-bar .badge {
    font-size: 0.7rem;
    background: #1a73e8;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 500;
}

/* Pipeline timeline */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 0;
    font-size: 0.88rem;
}
.step-icon {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    flex-shrink: 0;
    font-weight: 700;
}
.step-icon.done { background: #1a73e8; color: white; }
.step-icon.active { background: #f39c12; color: white; animation: pulse 1s infinite; }
.step-icon.warn { background: #e74c3c; color: white; }
.step-icon.pending { background: #333; color: #666; }

.step-content { flex: 1; }
.step-label {
    font-weight: 600;
    color: #e0e0e0;
    font-size: 0.88rem;
}
.step-detail {
    color: #888;
    font-size: 0.78rem;
    margin-top: 1px;
}
.step-time {
    color: #555;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Connector line between steps */
.pipeline-connector {
    width: 2px;
    height: 12px;
    background: #333;
    margin-left: 13px;
}

/* Confidence badge */
.conf-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.conf-badge.high { background: rgba(26,115,232,0.15); color: #4da6ff; border: 1px solid rgba(26,115,232,0.3); }
.conf-badge.med { background: rgba(243,156,18,0.15); color: #f0c040; border: 1px solid rgba(243,156,18,0.3); }
.conf-badge.low { background: rgba(231,76,60,0.15); color: #ff6b6b; border: 1px solid rgba(231,76,60,0.3); }

/* Answer card */
.answer-card {
    background: #141820;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 1.2rem;
    margin: 0.8rem 0;
}

/* Metric row */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 0.5rem 0;
}
.metric-item {
    background: #141820;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    flex: 1;
    text-align: center;
}
.metric-item .label { font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-item .value { font-size: 1.3rem; font-weight: 700; color: #e0e0e0; margin-top: 2px; }

/* Source chip */
.source-chip {
    background: #1a1d23;
    border: 1px solid #2a2d33;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
}
.source-chip .name { color: #4da6ff; font-weight: 600; }
.source-chip .score { color: #666; font-family: monospace; }

/* Claim verdict */
.verdict-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #1a1d23;
}
.verdict-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.verdict-dot.verified { background: #2ecc71; }
.verdict-dot.contradicted { background: #e74c3c; }
.verdict-dot.unverifiable { background: #f39c12; }

/* Override Streamlit defaults */
.stButton > button[kind="primary"] {
    background: #1a73e8;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
.stTextInput > div > div > input {
    border-radius: 6px;
    border: 1px solid #333;
    background: #141820;
}
</style>
"""


def inject_css():
    """Call at top of every page to inject custom CSS."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def top_bar(title: str, badge: str = ""):
    """Render clean top bar."""
    import streamlit as st
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="top-bar"><h1>{title}</h1>{badge_html}</div>',
        unsafe_allow_html=True,
    )


def render_pipeline_step(step, is_last=False):
    """Render a single pipeline step as HTML for the live timeline."""
    ICONS = {
        "encode": "E", "retrieve": "R", "reason": "G",
        "evaluate": "V", "calibrate": "C", "fetch_new": "F", "done": "✓",
    }
    step_type = step.step_type.value
    icon = ICONS.get(step_type, "•")
    
    if step_type == "done":
        css_class = "done"
    elif step_type == "calibrate":
        css_class = "warn"
    else:
        css_class = "done"

    time_str = f"{step.duration_ms:.0f}ms" if step.duration_ms > 0 else ""
    cycle_str = f" · Cycle {step.cycle}" if step.cycle > 0 else ""

    html = f"""
    <div class="pipeline-step">
        <div class="step-icon {css_class}">{icon}</div>
        <div class="step-content">
            <div class="step-label">{step_type.replace('_', ' ').title()}{cycle_str}</div>
            <div class="step-detail">{step.detail}</div>
        </div>
        <div class="step-time">{time_str}</div>
    </div>
    """
    if not is_last:
        html += '<div class="pipeline-connector"></div>'
    return html


def conf_badge(value: float) -> str:
    """Return confidence badge HTML."""
    if value >= 0.70:
        cls = "high"
    elif value >= 0.50:
        cls = "med"
    else:
        cls = "low"
    return f'<span class="conf-badge {cls}">{value:.3f}</span>'
