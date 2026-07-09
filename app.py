import uuid
from datetime import date as date_cls

import streamlit as st
from langgraph.types import Command

from interview_graph import build_app, make_initial_state

st.set_page_config(page_title="AI Interviewer", page_icon="🧑‍💼", layout="centered")

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
