"""
Aether — an autonomous research desk.

Streamlit front-end for the search -> read -> write -> critique pipeline
defined in src/pipeline.py.
"""
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import re
import sys
import html
import traceback
from datetime import datetime

import streamlit as st

from src.Agents.agents import build_search_agent, reader_agent, writer_chain, critic_chain
from src.tools.tools import search_web, scrape_url
from src.Pipeline.pipeline import run_research_pipeline  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aether — Research Desk",
    page_icon="🜂",
    layout="wide",
    initial_sidebar_state="expanded",
)

STAGES = [
    ("Scout", "searches the open web for recent, credible sources"),
    ("Reader", "opens the strongest source and extracts its full text"),
    ("Scribe", "drafts a structured report from what was found"),
    ("Critic", "scores the report and flags what's weak"),
]

SAMPLE_STATE = {
    "topic": "The economics of small modular nuclear reactors",
    "search_results": (
        "Title: Small Modular Reactors: A Technology Review\n"
        "URL: https://www.iea.org/reports/smr-technology-review\n"
        "Snippet: An overview of SMR designs, cost trajectories, and deployment timelines across major markets.\n"
        "----\n"
        "Title: Why SMR Costs Keep Slipping\n"
        "URL: https://www.utilitydive.com/news/smr-cost-overruns\n"
        "Snippet: A look at the gap between projected and realized costs for first-of-a-kind SMR projects.\n"
    ),
    "scraped_content": (
        "Small modular reactors (SMRs) are pitched as a way to bring nuclear power's low-carbon baseload capacity to grids..."
    ),
    "report": (
        "## Introduction\n\n"
        "Small modular reactors (SMRs) have moved from a niche engineering concept to a central plank of many countries' decarbonization plans. This report summarizes the current economic picture: what SMRs promise, what has actually shipped, and where the cost curve is likely headed.\n\n"
        "## Key Findings\n\n"
        "**1. Factory fabrication is the core cost thesis, not yet the reality.**\nVendors argue that building reactors on an assembly line, rather than pouring concrete on-site, is what will make SMRs cheaper per megawatt than large plants. So far, only a handful of first-of-a-kind units have been built, so this thesis is still unproven at scale.\n\n"
        "**2. First projects have overshot their budgets.**\nEarly SMR builds have generally cost more than forecast, echoing the cost history of large reactors. Regulatory review timelines, immature supply chains, and one-off engineering work all add cost that is supposed to disappear once designs are standardized and repeated.\n\n"
        "**3. The investment case depends heavily on policy support.**\nLoan guarantees, production tax credits, and government co-investment currently do most of the work of making SMR economics pencil out. Removing that support would push the breakeven point for most announced projects out by years.\n\n"
        "## Conclusion\n\n"
        "SMRs may still achieve the cost curve their backers promise, but the evidence so far supports caution rather than confidence: the technology is real, the economics are not yet proven, and near-term deployment remains dependent on public subsidy.\n\n"
        "## Sources\n\n"
        "- https://www.iea.org/reports/smr-technology-review\n"
        "- https://www.utilitydive.com/news/smr-cost-overruns\n"
    ),
    "critique": (
        "Score: 7/10\n\n"
        "Strengths:\n"
        "- Clear separation between the cost *thesis* and the cost *evidence*\n"
        "- Conclusion is appropriately hedged rather than overclaiming\n\n"
        "Areas to Improve:\n"
        "- Only two sources; a stronger report would triangulate across more\n"
        "- No discussion of specific projects or dollar figures\n\n"
        "One line Verdict:\n"
        "Solid, honest first draft that would benefit from harder numbers."
    ),
}

# ──────────────────────────────────────────────────────────────────────────
# Production-Level UI/UX CSS
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-dark: #090f0c;
  --bg-gradient: radial-gradient(130% 130% at 50% 0%, #15241e 0%, #090f0c 60%, #050806 100%);
  --ink: #0d1713;
  --ink-2: #121f1a;
  --panel: #16241f;
  --panel-border: #23362d;
  --paper: #f2ebd9;
  --paper-2: #e6dfcd;
  --brass: #d4af37;
  --brass-soft: #ebd078;
  --brass-hover: #f1c40f;
  --clay: #c96247;
  --sage: #95b8a8;
  --text: #f0f4f1;
  --text-muted: #8b9e95;
  --ink-text: #1a150d;
}

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--panel-border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--sage); }

.stApp {
  background: var(--bg-gradient);
  color: var(--text);
  background-attachment: fixed;
}

#MainMenu, footer, header[data-testid="stHeader"] { display: none; }

section[data-testid="stSidebar"] {
  background: rgba(18, 31, 26, 0.6) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-right: 1px solid var(--panel-border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 { 
    font-family: 'Fraunces', serif; 
    font-weight: 500; 
    letter-spacing: -0.02em; 
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(212, 175, 55, 0.4); }
    70% { box-shadow: 0 0 0 8px rgba(212, 175, 55, 0); }
    100% { box-shadow: 0 0 0 0 rgba(212, 175, 55, 0); }
}

/* Hero Section */
.aether-hero {
  padding: 3rem 0 1.5rem 0;
  border-bottom: 1px solid var(--panel-border);
  margin-bottom: 2rem;
  animation: fadeInUp 0.6s ease-out forwards;
}
.aether-mark {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  color: var(--sage);
  margin-bottom: 0.75rem;
  text-transform: uppercase;
}
.aether-title {
  font-family: 'Fraunces', serif;
  font-size: 3rem;
  font-weight: 500;
  color: var(--text);
  line-height: 1.1;
  margin: 0;
}
.aether-sub {
  color: var(--text-muted);
  max-width: 55ch;
  margin-top: 1rem;
  font-size: 1.1rem;
  line-height: 1.6;
  font-weight: 300;
}

/* Stage Tracker */
.stage-track { display: flex; flex-direction: column; gap: 0; margin-top: 0.8rem; }
.stage-row { display: flex; gap: 1rem; position: relative; padding-bottom: 1.6rem; }
.stage-row:last-child { padding-bottom: 0; }
.stage-rail {
  width: 2px; position: absolute; left: 6px; top: 18px; bottom: -2px;
  background: var(--panel-border);
  border-radius: 2px;
}
.stage-row:last-child .stage-rail { display: none; }
.stage-dot {
  width: 14px; height: 14px; border-radius: 50%;
  background: var(--bg-dark); border: 2px solid var(--panel-border);
  margin-top: 3px; flex-shrink: 0; z-index: 1;
  transition: all 0.3s ease;
}
.stage-dot.done { background: var(--sage); border-color: var(--sage); }
.stage-dot.active { 
    background: var(--brass); 
    border-color: var(--brass); 
    animation: pulseGlow 2s infinite;
}
.stage-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.05em; transition: color 0.3s ease;
}
.stage-label.on { color: var(--brass-soft); font-weight: 500; }
.stage-note { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.2rem; line-height: 1.4; }

/* Main Panels */
.st-key-report_panel {
  background: var(--paper);
  border-radius: 6px;
  border-top: 6px solid var(--brass);
  padding: 3.5rem 4rem;
  box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05);
  animation: fadeInUp 0.5s ease-out forwards;
}
.st-key-report_panel h1, .st-key-report_panel h2, .st-key-report_panel h3 {
  color: var(--ink-text);
}
.st-key-report_panel, .st-key-report_panel p, .st-key-report_panel li {
  color: #2c251a;
  font-size: 1.05rem;
  line-height: 1.75;
}
.st-key-report_panel h1 { font-size: 2.2rem; margin-bottom: 0.5rem; }
.st-key-report_panel h2 {
  font-size: 1.4rem;
  border-bottom: 1px solid #d8ccb0;
  padding-bottom: 0.5rem;
  margin-top: 2rem;
  margin-bottom: 1rem;
}
.st-key-report_panel strong { color: #1a150d; }

.st-key-critic_panel, .st-key-meta_panel {
  background: rgba(22, 36, 31, 0.6);
  backdrop-filter: blur(10px);
  border: 1px solid var(--panel-border);
  border-radius: 6px;
  padding: 2rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  animation: fadeInUp 0.6s ease-out forwards;
}
.st-key-critic_panel p, .st-key-critic_panel li { color: var(--text); line-height: 1.6; }
.critic-score {
  font-family: 'Fraunces', serif; font-size: 3.5rem; color: var(--brass-soft); line-height: 1; margin-top: 0.5rem;
}
.critic-score-label { 
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; 
    color: var(--sage); letter-spacing: 0.1em; text-transform: uppercase; 
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 1px dashed var(--panel-border);
  border-radius: 8px;
  padding: 5rem 2rem;
  text-align: center;
  color: var(--text-muted);
  background: rgba(18, 31, 26, 0.3);
  animation: fadeInUp 0.4s ease-out forwards;
}
.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}
.empty-state h3 { font-family: 'Inter', sans-serif; font-size: 1.2rem; color: var(--text); margin-bottom: 0.5rem; }

/* Inputs and Buttons */
.stTextArea textarea, .stTextInput input {
  background: rgba(9, 15, 12, 0.8) !important;
  color: var(--text) !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 4px !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--brass) !important;
    box-shadow: 0 0 0 1px var(--brass) !important;
}

.stButton > button {
  background: var(--brass) !important;
  color: #1a150d !important;
  border: none !important;
  border-radius: 4px !important;
  font-weight: 600 !important;
  padding: 0.6rem 1.2rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 12px rgba(212, 175, 55, 0.2) !important;
}
.stButton > button:hover { 
    background: var(--brass-hover) !important; 
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(212, 175, 55, 0.3) !important;
}
.stButton > button:active { transform: translateY(1px); }

.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.03) !important;
  color: var(--text) !important;
  border: 1px solid var(--panel-border) !important;
  box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.06) !important;
    border-color: var(--sage) !important;
}

[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  border: 1px solid var(--brass) !important;
  color: var(--brass-soft) !important;
  border-radius: 4px !important;
  transition: all 0.2s ease !important;
  margin-top: 1rem;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(212, 175, 55, 0.1) !important;
}

.error-card {
  border-left: 4px solid var(--clay);
  background: rgba(201, 98, 71, 0.08);
  padding: 1.2rem 1.5rem;
  border-radius: 0 4px 4px 0;
  font-size: 0.95rem;
  animation: fadeInUp 0.4s ease-out forwards;
}
.error-card code { color: var(--brass-soft); background: rgba(0,0,0,0.2); padding: 2px 6px; border-radius: 3px;}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "run_at" not in st.session_state:
    st.session_state.run_at = None
if "completed_step" not in st.session_state:
    st.session_state.completed_step = 0

def keys_configured() -> bool:
    return bool(os.getenv("TAVILY_API_KEY")) and bool(os.getenv("GEMINI_API_KEY"))

def parse_score(critique: str):
    match = re.search(r"score\s*:?\s*(\d+(?:\.\d+)?)\s*/\s*10", critique, re.IGNORECASE)
    return match.group(1) if match else None

def render_stage_tracker(container, current_step: int, messages: dict):
    """current_step: 0 = nothing started, 1-4 = that stage active, 5 = all done."""
    rows = []
    for i, (label, desc) in enumerate(STAGES, start=1):
        if current_step > i or current_step == 5:
            dot_class, label_class = "done", "on"
        elif current_step == i:
            dot_class, label_class = "active", "on"
        else:
            dot_class, label_class = "", ""
        note = messages.get(i, desc)
        rows.append(
            f"""<div class="stage-row">
                <div class="stage-rail"></div>
                <div class="stage-dot {dot_class}"></div>
                <div>
                    <div class="stage-label {label_class}">{html.escape(label)}</div>
                    <div class="stage-note">{html.escape(note)}</div>
                </div>
            </div>"""
        )
    container.markdown(f'<div class="stage-track">{"".join(rows)}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — the research desk
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### 🜂 Aether")
    st.caption("An autonomous research desk: four small agents, one dossier.")

    st.markdown("---")
    topic_input = st.text_area(
        "Research topic",
        placeholder="e.g. The competitive landscape for humanoid robotics in 2026",
        height=100,
    )

    run_clicked = st.button("Run Research", use_container_width=True)
    demo_clicked = st.button("Load Sample Dossier", use_container_width=True, type="secondary")

    st.markdown("---")
    st.markdown("**Pipeline Status**")
    tracker_slot = st.empty()
    render_stage_tracker(tracker_slot, st.session_state.completed_step, {})

    if not keys_configured():
        st.markdown("---")
        st.caption("⚠ `TAVILY_API_KEY` or `GEMINI_API_KEY` not detected. Sample dossier still functional.")

# ──────────────────────────────────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="aether-hero">
  <div class="aether-mark">Scout · Reader · Scribe · Critic</div>
  <p class="aether-title">The research desk that shows its work.</p>
  <p class="aether-sub">Give it a topic. Four agents search, read, write, and critique in sequence — delivering a comprehensive dossier rather than a basic chat response.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Run pipeline
# ──────────────────────────────────────────────────────────────────────────
if demo_clicked:
    st.session_state.result = SAMPLE_STATE
    st.session_state.topic = SAMPLE_STATE["topic"]
    st.session_state.run_at = datetime.now()
    st.session_state.completed_step = 5

elif run_clicked:
    if not topic_input.strip():
        st.warning("Enter a topic before running the pipeline.")
    elif not keys_configured():
        st.markdown(
            """
<div class="error-card">
<b>Missing API keys.</b> This pipeline needs <code>TAVILY_API_KEY</code> and
<code>GEMINI_API_KEY</code> set as environment variables (a <code>.env</code>
file in the project root works). Add them and rerun — or click
<b>"Load Sample Dossier"</b> in the sidebar to preview the interface.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        messages = {}

        # If your run_research_pipeline supports progress callbacks, use this:
        # def on_progress(step, label, message):
        #     st.session_state.completed_step = step
        #     messages[step] = message
        #     render_stage_tracker(tracker_slot, step, messages)
        # 
        # And change the call below to: 
        # result = run_research_pipeline(topic_input.strip(), progress_callback=on_progress)

        # Assuming it doesn't support callbacks currently based on your previous code:
        with st.spinner("Agents are researching..."):
            try:
                st.session_state.completed_step = 2 # Show active state while running
                render_stage_tracker(tracker_slot, st.session_state.completed_step, messages)
                
                result = run_research_pipeline(topic_input.strip())
                
                st.session_state.result = result
                st.session_state.topic = topic_input.strip()
                st.session_state.run_at = datetime.now()
                st.session_state.completed_step = 5
                render_stage_tracker(tracker_slot, st.session_state.completed_step, messages)
            except Exception as exc:  # noqa: BLE001
                st.session_state.completed_step = 0
                st.markdown(
                    f"""
<div class="error-card">
<b>The pipeline stopped partway through.</b><br/>
<code>{html.escape(str(exc))}</code><br/><br/>
Check that your API keys are valid and that the model name in
<code>src/Agents/agents.py</code> matches a model available to your account.
</div>
""",
                    unsafe_allow_html=True,
                )
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())

# ──────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────
result = st.session_state.result

if result is None:
    st.markdown(
        """
<div class="empty-state">
<div class="empty-state-icon">🜂</div>
<h3>Awaiting Directives</h3>
<p>Enter a topic in the sidebar and click <b>Run Research</b> — or load the sample dossier to see how a finished report looks.</p>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    report_col, side_col = st.columns([2.2, 1], gap="large")

    with report_col:
        with st.container(key="report_panel"):
            st.markdown(f"# {result['topic']}")
            if st.session_state.run_at:
                st.caption(f"Compiled on {st.session_state.run_at.strftime('%B %d, %Y at %H:%M')}")
            st.markdown(result["report"])

        st.download_button(
            "Download Report (Markdown)",
            data=f"# {result['topic']}\n\n{result['report']}",
            file_name="research_report.md",
            mime="text/markdown",
        )

    with side_col:
        score = parse_score(result["critique"])
        with st.container(key="critic_panel"):
            st.markdown('<div class="critic-score-label">Critic\'s Score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="critic-score">{score or "—"}<span style="font-size:1.4rem;color:var(--text-muted);">/10</span></div>', unsafe_allow_html=True)
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.markdown(result["critique"])

        st.write("")
        with st.container(key="meta_panel"):
            st.markdown("### Sources Consulted")
            urls = re.findall(r"https?://\S+", result.get("search_results", ""))
            urls = list(dict.fromkeys(urls))
            if urls:
                for u in urls:
                    # Clean up trailing markdown characters if any
                    clean_u = u.rstrip('")].*')
                    st.markdown(f"- [{clean_u}]({clean_u})")
            else:
                st.caption("No URLs detected in the search stage output.")

        st.write("")
        with st.expander("View Raw Search Data"):
            st.text(result.get("search_results", "No search results available."))
        with st.expander("View Extracted Text"):
            st.text(result.get("scraped_content", "No scraped content available."))