"""Streamlit chat interface."""
import os
import uuid
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from src.path_setup import ensure_project_root
else:
    from path_setup import ensure_project_root

ensure_project_root()

from src.assistant import ConstructionAssistant
from src.chat_ui import CHATBOT_NAME, THINKING_LABEL, auto_scroll_script, chat_bubble_css, client_rate_limit_key, get_visible_suggestions, greeting_for_hour, select_prompt, should_show_suggestions, startup_failure_message, submit_turn
from src.diagnostics import log_debug_turn
from src.leads import prune_expired_leads
from src.presentation import format_answer
from src.rag import HybridKnowledgeBase
from src.settings import load_streamlit_secrets

logger = logging.getLogger(__name__)


load_dotenv()
try:
    load_streamlit_secrets(os.environ, st.secrets)
except StreamlitSecretNotFoundError:
    # Streamlit raises when no secrets file/config is present; local .env remains valid.
    pass
st.set_page_config(page_title=f"{CHATBOT_NAME} | Construction & Interiors", page_icon="🤖", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []
if "lead" not in st.session_state:
    st.session_state.lead = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "used_suggestions" not in st.session_state:
    st.session_state.used_suggestions = set()

with st.container(key="app_header"):
    title_col, action_col = st.columns([5, 1.5], vertical_alignment="center")
    with title_col:
        st.title(f"🤖 {CHATBOT_NAME}")
    with action_col:
        if st.button("＋ New chat", key="new_conversation_top", use_container_width=True):
            st.session_state.history = []
            st.session_state.lead = {}
            st.session_state.session_id = uuid.uuid4().hex
            st.session_state.used_suggestions = set()
            st.rerun()
    st.caption("Construction & Interiors AI Chatbot")
st.markdown(chat_bubble_css(), unsafe_allow_html=True)


@st.cache_resource
def services():
    kb = HybridKnowledgeBase(
        os.getenv("DATA_DIR", "data"),
        os.getenv("CHROMA_DIR", "chroma_db"),
        int(os.getenv("CHUNK_SIZE", "900")),
        int(os.getenv("CHUNK_OVERLAP", "140")),
    )
    kb.ingest()
    if os.getenv("LEADS_ENCRYPTION_KEY"):
        prune_expired_leads(
            os.getenv("LEADS_FILE", "leads.enc.jsonl"),
            int(os.getenv("LEADS_RETENTION_DAYS", "180")),
        )
    return kb, ConstructionAssistant(kb, os.getenv("LEADS_FILE", "leads.enc.jsonl"))


try:
    _, assistant = services()
except Exception:
    logger.exception("Knowledge base startup failed")
    st.error(startup_failure_message())
    st.stop()

if not os.getenv("LEADS_ENCRYPTION_KEY"):
    st.sidebar.warning("Lead encryption is not configured. Lead details will not be saved until LEADS_ENCRYPTION_KEY is set.")
if getattr(assistant, "general_client_error", None):
    st.sidebar.warning("Gemini general answers are unavailable because the Google GenAI SDK could not be imported. Check the app dependency install logs and redeploy.")

history = st.session_state.history
with st.container(border=True, key="chat_panel"):
    if not history:
        india_time = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        welcome_message = greeting_for_hour(india_time.hour)
        st.session_state.history = [{"role": "assistant", "content": welcome_message}]
        history = st.session_state.history

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(format_answer(message["content"]))

    suggestions = [
        "What warranty do you give on modular kitchens?",
        "What is the premium interior rate per sq. ft.?",
        "Mere bathroom renovation mein kitna time lagega?",
        "I have a 1,200 sq. ft. flat. What could a full interior cost?",
        "Compare standard and premium packages for a 2BHK.",
        "How long will the project take, and what if there is a delay?",
        "Which materials and brands do you use for modular kitchens?",
        "What are the payment milestones for a project?",
        "I want to start next month, and my budget is around 15 lakhs.",
        "What happens during the first site survey?",
        "Which cities do you currently serve?",
        "What warranty do you give on wardrobes?",
    ]
    clicked = []
    clicked_suggestion = None
    visible_suggestions = get_visible_suggestions(suggestions, st.session_state.used_suggestions, count=4)
    if visible_suggestions and should_show_suggestions(history):
        st.markdown("**You can also ask**")
        suggestion_columns = st.columns(2)
        for index, suggestion in enumerate(visible_suggestions):
            with suggestion_columns[index % 2]:
                is_clicked = st.button(suggestion, key=f"suggestion_{suggestions.index(suggestion)}", use_container_width=True)
                clicked.append(suggestion if is_clicked else None)
                if is_clicked:
                    clicked_suggestion = suggestion

    typed_prompt = st.chat_input("Ask in English or Hindi, or tell us about your project…")
    prompt = select_prompt(typed_prompt, clicked)
    if prompt and len(prompt) > 12000:
        st.error("Your message is too long. Please keep it under 12,000 characters.")
    elif prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            with st.chat_message("assistant"):
                with st.spinner(THINKING_LABEL):
                    new_history, new_lead, answer = submit_turn(
                        assistant, prompt, st.session_state.history, st.session_state.lead,
                        client_rate_limit_key(
                            getattr(getattr(st, "context", None), "ip_address", None),
                            st.session_state.session_id,
                        ),
                    )
            st.session_state.history = new_history
            st.session_state.lead = new_lead
            log_debug_turn(prompt, answer, st.session_state.session_id, new_lead, assistant.last_retrieval)
            if clicked_suggestion:
                st.session_state.used_suggestions.add(clicked_suggestion)
            st.rerun()
        except Exception as exc:
            with st.chat_message("assistant"):
                st.error(f"I couldn't complete that reply just now. Please try again. ({type(exc).__name__})")

st.html(auto_scroll_script(), unsafe_allow_javascript=True)
