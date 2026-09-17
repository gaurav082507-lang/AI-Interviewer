import uuid
from datetime import date as date_cls

import streamlit as st
from langgraph.types import Command

from interview_graph import build_app, make_initial_state

st.set_page_config(page_title="AI Interviewer", page_icon="🧑‍💼", layout="centered")

# ---------------------------------------------------------------------------
# Dark gradient theme (matches the Interviewer AI engine styling)
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at 50% -20%, #1c1d3a 0%, #070812 55%, #04050a 100%);
            color: #e2e8f0;
            font-family: 'Inter', system-ui, sans-serif;
        }

        [data-testid="stSidebar"] {
            background-color: #0a0b15;
            border-right: 1px solid #151830;
        }

        h1, h2, h3 {
            color: #ffffff !important;
            letter-spacing: -0.5px;
        }

        .stApp h1 {
            background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        [data-testid="stForm"] {
            background: linear-gradient(180deg, rgba(22, 26, 51, 0.55) 0%, rgba(11, 13, 26, 0.65) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5) !important;
            border-radius: 14px !important;
            padding: 26px !important;
        }

        .stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
            background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%);
            color: #ffffff !important;
            border: none;
            font-weight: 600;
            border-radius: 6px;
            padding: 10px 24px;
            transition: all 0.2s ease;
        }
        .stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover {
            box-shadow: 0 0 18px rgba(79, 70, 229, 0.4);
            transform: translateY(-1px);
        }

        div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="datepicker"] {
            background-color: #0e101f !important;
            border: 1px solid #1f2342 !important;
            border-radius: 6px !important;
        }
        label p {
            color: #94a3b8 !important;
            font-size: 14px !important;
            font-weight: 500 !important;
        }

        [data-testid="stChatMessage"] {
            background: rgba(18, 22, 43, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
        }

        .stAlert {
            background: rgba(16, 185, 129, 0.08) !important;
            border: 1px solid rgba(16, 185, 129, 0.25) !important;
        }

        .streamlit-expanderHeader {
            background: rgba(18, 22, 43, 0.4) !important;
            border-radius: 8px !important;
            color: #e2e8f0 !important;
        }

        [data-testid="stCaptionContainer"] {
            color: #7f8ea6 !important;
        }

        footer, header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "app" not in st.session_state:
    st.session_state.app = build_app()

if "started" not in st.session_state:
    st.session_state.started = False

if "finished" not in st.session_state:
    st.session_state.finished = False

if "history" not in st.session_state:
    st.session_state.history = []  # list of ("ai"/"human", text)

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def process_result(result):
    """Inspect the graph's return value and update session state accordingly."""
    if "__interrupt__" in result:
        question = result["__interrupt__"][0].value["Question"]
        st.session_state.current_question = question
        st.session_state.history.append(("ai", question))
    else:
        # Graph reached END -> final message is the evaluation string
        evaluation = result.get("interview_question", "")
        st.session_state.evaluation = evaluation
        st.session_state.finished = True
        st.session_state.current_question = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧑‍💼 AI Interview System")
    st.write(
        "This is a conversational AI interviewer. Enter the role you're "
        "interviewing for, click **Start Interview**, and answer the "
        "questions one at a time in the chat box below."
    )
    if st.session_state.started and not st.session_state.finished:
        if st.button("🔄 Restart Interview"):
            st.session_state.clear()
            st.rerun()

    st.divider()
    st.markdown(
        """
        **Made By Gaurav Gupta**
        [LinkedIn Profile](https://www.linkedin.com/in/gaurav-gupta-79754a377)
        """
    )

st.title("🧑‍💼 AI Interview System")

# ---------------------------------------------------------------------------
# Pre-interview setup screen
# ---------------------------------------------------------------------------
if not st.session_state.started:
    st.subheader("Set up your interview")
    with st.form("setup_form"):
        role = st.text_input("Role you're applying for", placeholder="e.g. Junior ML Engineer")
        interview_date = st.date_input("Interview date", value=date_cls.today())
        submitted = st.form_submit_button("🚀 Start Interview")

    if submitted:
        if not role.strip():
            st.error("Please enter a role before starting.")
        else:
            st.session_state.role = role.strip()
            st.session_state.date = str(interview_date)
            st.session_state.thread_id = str(uuid.uuid4())

            initial_state = make_initial_state(st.session_state.role, st.session_state.date)
            result = st.session_state.app.invoke(initial_state, config=get_config())
            process_result(result)

            st.session_state.started = True
            st.rerun()

# ---------------------------------------------------------------------------
# Interview in progress
# ---------------------------------------------------------------------------
elif st.session_state.started and not st.session_state.finished:
    st.caption(f"Interviewing for: **{st.session_state.role}**  |  Date: {st.session_state.date}")

    for speaker, text in st.session_state.history:
        role_label = "assistant" if speaker == "ai" else "user"
        with st.chat_message(role_label):
            st.write(text)

    answer = st.chat_input("Type your answer here...")
    if answer:
        st.session_state.history.append(("human", answer))
        result = st.session_state.app.invoke(Command(resume=answer), config=get_config())
        process_result(result)
        st.rerun()

# ---------------------------------------------------------------------------
# Interview finished -> show evaluation
# ---------------------------------------------------------------------------
else:
    st.success("✅ Interview completed!")

    with st.expander("📜 View full conversation", expanded=False):
        for speaker, text in st.session_state.history:
            role_label = "assistant" if speaker == "ai" else "user"
            with st.chat_message(role_label):
                st.write(text)

    st.subheader("📋 Evaluation Report")
    st.text(st.session_state.evaluation)

    st.download_button(
        "⬇️ Download Evaluation as .txt",
        data=st.session_state.evaluation or "",
        file_name=f"evaluation_{st.session_state.role.replace(' ', '_')}.txt",
        mime="text/plain",
    )

    if st.button("🔄 Start a New Interview"):
        st.session_state.clear()
        st.rerun()
