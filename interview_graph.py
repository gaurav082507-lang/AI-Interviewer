import uuid
import datetime as dt

import streamlit as st
from langgraph.types import Command

from interview_graph import build_app, make_initial_state

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Panelist AI · Autonomous Interview Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme — dark navy canvas, blue → violet gradient accent, mono labels
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root{
        --bg-0:#080b13;
        --bg-1:#0c1120;
        --panel:#0e1526;
        --panel-2:#111a2e;
        --border:#1e293b;
        --border-soft:#182236;
        --text-hi:#e7ecf7;
        --text-mid:#aab4c8;
        --text-low:#6b7690;
        --accent-1:#5b8cff;
        --accent-2:#8a7dff;
        --accent-3:#b16cff;
        --green:#34d399;
        --grad: linear-gradient(100deg, var(--accent-1), var(--accent-2) 55%, var(--accent-3));
    }

    html, body, [class*="css"]{
        font-family:'Inter', sans-serif;
    }

    .stApp{
        background:
            radial-gradient(1100px 600px at 12% -8%, rgba(91,140,255,0.10), transparent 55%),
            radial-gradient(900px 500px at 100% 0%, rgba(177,108,255,0.08), transparent 50%),
            var(--bg-0);
        color: var(--text-hi);
    }

    #MainMenu, footer {visibility:hidden;}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"]{
        background: var(--bg-1);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] .block-container{
        padding-top: 2rem;
    }

    .brand-row{
        display:flex; align-items:center; gap:10px; margin-bottom:2px;
    }
    .brand-icon{
        width:34px; height:34px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        background: var(--grad); font-size:17px;
        box-shadow: 0 0 24px rgba(90,120,255,0.35);
    }
    .brand-name{
        font-weight:800; font-size:1.05rem; letter-spacing:-0.01em; color:var(--text-hi);
    }
    .brand-sub{
        font-family:'JetBrains Mono', monospace;
        font-size:0.74rem; color:var(--text-mid);
        margin: 10px 0 16px 0; line-height:1.5;
    }

    .status-pill{
        display:inline-flex; align-items:center; gap:8px;
        font-family:'JetBrains Mono', monospace; font-size:0.74rem;
        color: var(--green);
        background: rgba(52,211,153,0.08);
        border: 1px solid rgba(52,211,153,0.35);
        padding: 7px 12px; border-radius:999px;
        margin-bottom: 18px;
    }
    .status-dot{
        width:7px; height:7px; border-radius:50%;
        background: var(--green);
        box-shadow: 0 0 0 0 rgba(52,211,153,0.6);
        animation: pulse 2s infinite;
    }
    @keyframes pulse{
        0%   { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }
        70%  { box-shadow: 0 0 0 7px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }

    .sidebar-label{
        font-family:'JetBrains Mono', monospace; font-size:0.72rem;
        letter-spacing:0.06em; text-transform:uppercase;
        color: var(--text-mid); margin: 18px 0 8px 0;
        display:flex; align-items:center; gap:6px;
    }

    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTextArea"] textarea{
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-hi) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stDateInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus{
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 0 1px var(--accent-1) !important;
    }
    label, .stMarkdown p { color: var(--text-mid); }

    /* Buttons */
    .stButton>button{
        width:100%;
        background: var(--grad);
        color: #060812;
        font-weight:700;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1rem;
        box-shadow: 0 8px 26px rgba(90,120,255,0.28);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton>button:hover{
        transform: translateY(-1px);
        box-shadow: 0 10px 30px rgba(120,110,255,0.4);
        color:#060812;
    }
    .stButton>button:disabled{
        opacity:0.5;
    }
    .stDownloadButton>button{
        width:100%;
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text-hi);
        border-radius: 12px;
        font-weight:600;
    }

    /* ---------- Hero ---------- */
    .hero-wrap{ padding: 8px 0 6px 0; }
    .hero-eyebrow{
        font-family:'JetBrains Mono', monospace;
        letter-spacing: 0.32em; font-size:0.78rem;
        color: var(--text-mid); text-align:center;
        margin-bottom: 14px;
    }
    .hero-title{
        text-align:center;
        font-weight:800; font-size: 3.2rem; letter-spacing:-0.03em;
        margin: 0 0 14px 0;
        background: var(--grad);
        background-size: 220% 220%;
        -webkit-background-clip: text; background-clip:text; color: transparent;
        animation: shift 8s ease infinite;
    }
    @keyframes shift{
        0%{background-position:0% 50%;}
        50%{background-position:100% 50%;}
        100%{background-position:0% 50%;}
    }
    .hero-sub{
        text-align:center; color: var(--text-mid);
        font-size: 1.05rem; max-width: 720px; margin: 0 auto 26px auto; line-height:1.6;
    }

    .badge-row{ display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-bottom: 30px; }
    .badge-pill{
        font-family:'JetBrains Mono', monospace; font-size:0.82rem;
        color: var(--text-hi);
        background: var(--panel);
        border: 1px solid var(--border);
        padding: 9px 16px; border-radius:999px;
        display:flex; align-items:center; gap:8px;
    }

    .callout{
        background: var(--panel);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 20px 24px;
        text-align:center;
        color: var(--text-mid);
        font-size: 1rem;
        max-width: 760px;
        margin: 0 auto;
    }
    .callout b{ color: var(--text-hi); }

    /* ---------- Chat transcript ---------- */
    .chat-wrap{ max-width: 780px; margin: 10px auto 0 auto; }
    .msg{
        border-radius: 14px; padding: 14px 18px; margin-bottom: 14px;
        line-height:1.55; font-size:0.96rem;
        animation: fadein 0.35s ease;
        border: 1px solid var(--border-soft);
    }
    @keyframes fadein{
        from{ opacity:0; transform: translateY(6px); }
        to{ opacity:1; transform: translateY(0); }
    }
    .msg-ai{
        background: var(--panel);
        color: var(--text-hi);
        border-color: var(--border);
    }
    .msg-ai .tag{ color: var(--accent-2); }
    .msg-human{
        background: linear-gradient(120deg, rgba(91,140,255,0.14), rgba(177,108,255,0.10));
        color: var(--text-hi);
        border-color: rgba(120,120,255,0.25);
        margin-left: 40px;
    }
    .msg-human .tag{ color: var(--green); }
    .tag{
        font-family:'JetBrains Mono', monospace; font-size:0.7rem;
        letter-spacing:0.08em; text-transform:uppercase;
        display:block; margin-bottom:6px; opacity:0.8;
    }

    /* ---------- Report ---------- */
    .report-box{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 26px 28px;
        max-width: 780px; margin: 10px auto 0 auto;
        font-family:'JetBrains Mono', monospace;
        font-size: 0.86rem; line-height:1.75; color: var(--text-hi);
        white-space: pre-wrap;
    }
    .section-title{
        text-align:center; font-weight:700; font-size:1.3rem; color:var(--text-hi);
        margin: 6px 0 4px 0;
    }
    .section-sub{
        text-align:center; color: var(--text-low); font-size:0.85rem; margin-bottom:22px;
        font-family:'JetBrains Mono', monospace;
    }

    hr{ border-color: var(--border-soft); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
if "graph_app" not in st.session_state:
    st.session_state.graph_app = build_app()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "phase" not in st.session_state:
    st.session_state.phase = "setup"          # setup -> interviewing -> done
if "transcript" not in st.session_state:
    st.session_state.transcript = []           # [{"role": "ai"/"human", "text": ...}]
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "report" not in st.session_state:
    st.session_state.report = ""


def _config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _advance(resume_value=None, role=None, date=None):
    """Runs the graph forward until the next interrupt (question) or END (report)."""
    app = st.session_state.graph_app
    cfg = _config()
    if resume_value is None:
        app.invoke(make_initial_state(role, date), config=cfg)
    else:
        app.invoke(Command(resume=resume_value), config=cfg)

    snapshot = app.get_state(cfg)
    if snapshot.next:
        # paused at the Human node waiting for an answer
        interrupt_payload = snapshot.tasks[0].interrupts[0].value
        question = interrupt_payload.get("Question", "")
        st.session_state.current_question = question
        st.session_state.transcript.append({"role": "ai", "text": question})
        st.session_state.phase = "interviewing"
    else:
        # graph finished — final report is sitting in interview_question
        report_text = snapshot.values.get("interview_question", "")
        st.session_state.report = report_text
        st.session_state.phase = "done"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-icon">🎙️</div>
            <div class="brand-name">Panelist AI</div>
        </div>
        <div class="brand-sub">structured Q&A · LangGraph interrupts · Mistral scoring</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="status-pill"><span class="status-dot"></span>Mistral API key connected</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">📋 Interview setup</div>', unsafe_allow_html=True)

    role = st.text_input("Role", placeholder="e.g. Senior Backend Engineer", disabled=st.session_state.phase != "setup")
    date_val = st.date_input("Interview date", value=dt.date.today(), disabled=st.session_state.phase != "setup")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.phase == "setup":
        start_disabled = not role.strip()
        if st.button("🚀 Start Interview", disabled=start_disabled):
            with st.spinner("Preparing the first question…"):
                _advance(role=role.strip(), date=str(date_val))
            st.rerun()
    elif st.session_state.phase == "interviewing":
        st.button("🚀 Interview in progress…", disabled=True)
    else:
        st.button("✅ Interview complete", disabled=True)

    if st.session_state.phase != "setup":
        st.markdown("---")
        if st.button("🔄 Start a new interview"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.phase = "setup"
            st.session_state.transcript = []
            st.session_state.current_question = ""
            st.session_state.report = ""
            st.rerun()

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
if st.session_state.phase == "setup":
    st.markdown('<div class="hero-eyebrow">AUTONOMOUS&nbsp;&nbsp;INTERVIEW&nbsp;&nbsp;ENGINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Panelist AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Give it a role and a date, and it runs the whole structured interview — '
        'one question at a time, adaptive follow-ups, and a full AI-scored evaluation report at the end.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="badge-row">
            <div class="badge-pill">🕸️ LangGraph</div>
            <div class="badge-pill">🧭 Human-in-the-loop</div>
            <div class="badge-pill">🤖 Mistral</div>
            <div class="badge-pill">📊 Structured report</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="callout">👈 Enter a <b>role</b> and <b>date</b> in the sidebar, then click '
        '<b>Start Interview</b> to begin.</div>',
        unsafe_allow_html=True,
    )

elif st.session_state.phase == "interviewing":
    st.markdown('<div class="hero-eyebrow">LIVE&nbsp;&nbsp;INTERVIEW</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{role or "Candidate"} Interview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{date_val}</div>', unsafe_allow_html=True)

    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for turn in st.session_state.transcript:
        if turn["role"] == "ai":
            st.markdown(
                f'<div class="msg msg-ai"><span class="tag">Interviewer</span>{turn["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="msg msg-human"><span class="tag">You</span>{turn["text"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_area("Your answer", placeholder="Type your answer here…", label_visibility="collapsed", height=110)
        submitted = st.form_submit_button("💬 Send Answer")

    if submitted and answer.strip():
        st.session_state.transcript.append({"role": "human", "text": answer.strip()})
        with st.spinner("Thinking of the next question…"):
            _advance(resume_value=answer.strip())
        st.rerun()

else:  # done
    st.markdown('<div class="hero-eyebrow">EVALUATION&nbsp;&nbsp;COMPLETE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Interview Report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{role or "Candidate"} · {date_val}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-box">{st.session_state.report}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            "⬇️ Download report (.txt)",
            data=st.session_state.report,
            file_name=f"interview_report_{st.session_state.thread_id[:8]}.txt",
            mime="text/plain",
        )
