"""
Agentic AI Assistant — Streamlit front-end
===========================================
A LangChain ReAct agent (Gemini LLM) with Tavily search + WeatherStack tools,
wrapped in a clean, production-styled chat interface.

Configuration:
    All API keys are read ONLY from environment variables / a local .env file.
    Nothing secret is ever rendered in the UI — the sidebar only shows a
    connected / not-connected status per service.

    Required environment variables (put these in a .env file next to this
    script, or export them in your shell / hosting platform):
        GEMINI_API_KEY
        TAVILY_API_KEY
        WEATHER_STACK_API

Run with:
    streamlit run app.py
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langsmith import Client

load_dotenv()

# ==========================================
# 0. SECRETS — read-only, never displayed
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEATHER_STACK_API = os.getenv("WEATHER_STACK_API", "")

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. DESIGN SYSTEM — global CSS
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-canvas: #F6F7F9;
        --bg-surface: #FFFFFF;
        --border-subtle: #E4E7EC;
        --border-strong: #D0D5DD;
        --text-primary: #1B1F27;
        --text-secondary: #667085;
        --text-tertiary: #98A2B3;
        --accent: #3E6BF0;
        --accent-hover: #2F57D6;
        --accent-wash: #EEF2FF;
        --success: #12805C;
        --danger: #D0342C;
        --radius-sm: 8px;
        --radius-md: 14px;
        --gradient: linear-gradient(135deg, #4C8DF6 0%, #8C6CE0 55%, #E4699F 100%);
    }

    /* Force a light color scheme regardless of the user's OS/browser theme —
       this is what was leaking through as dark widgets (selectbox, bottom
       input bar) sitting inside an otherwise light page. */
    :root { color-scheme: light; }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
        color: var(--text-primary);
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: var(--text-primary);
    }

    /* Remove Streamlit chrome for a production feel */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { padding-top: 1.2rem; max-width: 900px; }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background: var(--bg-canvas) !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    .brand-mark {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 4px;
    }
    .brand-dot {
        width: 26px; height: 26px; border-radius: 8px;
        background: var(--gradient);
        flex-shrink: 0;
    }
    .brand-name { font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em; color: var(--text-primary); }
    .brand-sub { color: var(--text-tertiary) !important; font-size: 0.8rem; margin: 0 0 1.4rem 36px; }

    /* ---------- Native widgets: selects, sliders, inputs, toggles ---------- */
    div[data-baseweb="select"] > div {
        background: var(--bg-surface) !important;
        border-color: var(--border-strong) !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background: var(--bg-surface) !important;
    }
    div[data-baseweb="menu"] li, ul[role="listbox"] li {
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-strong) !important;
    }
    [data-testid="stSlider"] label, [data-testid="stNumberInput"] label,
    [data-testid="stToggle"] label, [data-testid="stSelectbox"] label {
        color: var(--text-primary) !important;
    }

    /* Inline code chips inside markdown/captions (avoid dark code-chip clash) */
    [data-testid="stMarkdownContainer"] code, [data-testid="stCaptionContainer"] code {
        background: var(--accent-wash) !important;
        color: var(--accent-hover) !important;
        padding: 1px 5px;
        border-radius: 4px;
    }

    .status-row {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.85rem; color: var(--text-secondary);
        padding: 6px 0;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .status-on { background: var(--success); }
    .status-off { background: var(--danger); }

    .sidebar-label {
        font-size: 0.72rem; color: var(--text-tertiary);
        font-weight: 600; margin: 1.4rem 0 0.4rem 0;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-subtle);
        background: var(--bg-surface);
        color: var(--text-primary);
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.5rem 0.9rem;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--accent-wash);
    }

    /* ---------- Header ---------- */
    .app-header {
        display: flex; align-items: center; gap: 12px;
        padding: 4px 0 18px 0;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 8px;
    }
    .app-header .brand-dot { width: 30px; height: 30px; border-radius: 9px; }
    .app-header-text h1 {
        font-size: 1.15rem; font-weight: 600; margin: 0; letter-spacing: -0.01em;
    }
    .app-header-text p { font-size: 0.82rem; color: var(--text-tertiary); margin: 0; }

    /* ---------- Empty state ---------- */
    .empty-state { text-align: center; padding: 4.5rem 1rem 2rem 1rem; }
    .empty-state .brand-dot { width: 44px; height: 44px; border-radius: 14px; margin: 0 auto 1.1rem auto; }
    .empty-state h2 {
        font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
    }
    .empty-state p { color: var(--text-secondary); font-size: 0.95rem; margin: 0; }

    /* ---------- Chat messages ---------- */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.35rem 0;
        border: none;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 0.94rem;
        line-height: 1.55;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: var(--accent-wash);
        border-radius: var(--radius-md);
        padding: 0.7rem 1rem;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 0.7rem 1rem;
    }

    /* ---------- Reasoning / tool trace ---------- */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        background: var(--bg-surface);
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        border-radius: 999px;
        border: 1px solid var(--border-strong);
        background: var(--bg-surface);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px var(--accent-wash);
    }

    /* ---------- Suggestion chips ---------- */
    .chip-row .stButton > button {
        border-radius: 999px;
        padding: 0.55rem 1rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 3. SIDEBAR — connection status & settings (no secrets shown)
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">
            <div class="brand-dot"></div>
            <div class="brand-name">Assistant</div>
        </div>
        <div class="brand-sub">Search &amp; weather agent</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="sidebar-label">CONNECTIONS</div>', unsafe_allow_html=True)

    def status_row(label, is_connected):
        dot_class = "status-on" if is_connected else "status-off"
        text = "Connected" if is_connected else "Not configured"
        st.markdown(
            f'<div class="status-row"><span class="status-dot {dot_class}"></span>'
            f'{label} — {text}</div>',
            unsafe_allow_html=True,
        )

    status_row("Gemini", bool(GEMINI_API_KEY))
    status_row("Tavily search", bool(TAVILY_API_KEY))
    status_row("WeatherStack", bool(WEATHER_STACK_API))

    if not (GEMINI_API_KEY and TAVILY_API_KEY and WEATHER_STACK_API):
        st.caption(
            "Missing keys are read from environment variables. Add them to a "
            "local .env file (GEMINI_API_KEY, TAVILY_API_KEY, WEATHER_STACK_API) "
            "and restart the app."
        )

    st.markdown('<div class="sidebar-label">MODEL</div>', unsafe_allow_html=True)
    model_name = "gemini-3.6-flash"
    st.markdown(
        '<div class="status-row"><span class="status-dot status-on"></span>'
        'gemini-3.6-flash</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Advanced"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.number_input("Max output tokens", 128, 8192, 1086, 64)
        show_reasoning = st.toggle("Show agent reasoning", value=False)

# ==========================================
# 4. HEADER
# ==========================================
st.markdown(
    """
    <div class="app-header">
        <div class="brand-dot"></div>
        <div class="app-header-text">
            <h1>Assistant</h1>
            <p>Answers questions using live web search and current weather data</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 5. WEATHER TOOL
# ==========================================
@tool
def get_weather_tool(city: str) -> str:
    """Fetch the current weather for a given city."""
    if not WEATHER_STACK_API:
        return "WeatherStack is not configured."

    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key={WEATHER_STACK_API}&query={city}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except requests.RequestException as e:
        return f"Error fetching weather for {city}: {e}"

    if "current" not in data:
        error_info = data.get("error", {}).get("info", "unknown error")
        return f"Could not fetch weather data for {city}: {error_info}"

    current = data["current"]
    return (
        f"City: {city}\n"
        f"Temperature: {current['temperature']}°C\n"
        f"Weather: {current['weather_descriptions'][0]}\n"
        f"Humidity: {current['humidity']}%"
    )


# ==========================================
# 6. AGENT BUILDER (cached)
# ==========================================
@st.cache_resource(show_spinner=False)
def build_agent_executor(_gemini_key, _tavily_key, _model_name, _temperature, _max_tokens):
    search_tool = TavilySearchResults(max_results=2)
    tools = [search_tool, get_weather_tool]

    llm = ChatGoogleGenerativeAI(
        model=_model_name,
        api_key=_gemini_key,
        max_tokens=_max_tokens,
        temperature=_temperature,
    )

    client = Client()
    prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


# ==========================================
# 7. CHAT STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

SUGGESTIONS = [
    "Find the capital of India and its current weather",
    "What's the weather in Tokyo right now?",
    "Search for the latest news on renewable energy",
]

def run_turn(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_area:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="✨"):
            output = None
            if not GEMINI_API_KEY:
                output = "Gemini isn't configured. Set GEMINI_API_KEY in your environment and restart the app."
            else:
                try:
                    agent_executor = build_agent_executor(
                        GEMINI_API_KEY, TAVILY_API_KEY, model_name, temperature, max_tokens
                    )
                    if show_reasoning:
                        with st.expander("Show reasoning", expanded=False):
                            st_callback = StreamlitCallbackHandler(st.container())
                            result = agent_executor.invoke(
                                {"input": user_input}, {"callbacks": [st_callback]}
                            )
                    else:
                        with st.spinner("Thinking…"):
                            result = agent_executor.invoke({"input": user_input})
                    output = result["output"]
                except Exception as e:
                    output = f"Something went wrong while running the agent: {e}"
            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})


# ==========================================
# 8. LAYOUT — empty state or chat history
# ==========================================
chat_area = st.container()

with chat_area:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="brand-dot"></div>
                <h2>What can I help you find?</h2>
                <p>Ask about current events, facts, or the weather anywhere.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chip-row">', unsafe_allow_html=True)
        cols = st.columns(len(SUGGESTIONS))
        for col, suggestion in zip(cols, SUGGESTIONS):
            with col:
                if st.button(suggestion, use_container_width=True, key=f"chip-{suggestion}"):
                    run_turn(suggestion)
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            avatar = "✨" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

# ==========================================
# 9. INPUT
# ==========================================
user_input = st.chat_input("Message Assistant…")
if user_input:
    run_turn(user_input)
    st.rerun()