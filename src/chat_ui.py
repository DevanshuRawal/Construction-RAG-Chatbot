"""Small, testable helpers for dispatching chat input from the Streamlit UI."""

import hashlib
import ipaddress

CHATBOT_NAME = "BuildMate AI"


def client_rate_limit_key(client_ip, session_id):
    """Share request limits across sessions from the same client without storing its IP."""
    if client_ip:
        try:
            normalized_ip = ipaddress.ip_address(str(client_ip).strip()).compressed
        except ValueError:
            normalized_ip = str(client_ip).strip()
        if normalized_ip:
            digest = hashlib.sha256(normalized_ip.encode("utf-8")).hexdigest()
            return f"client:{digest}"
    return f"session:{session_id}"


def startup_failure_message():
    """Return a safe user-facing startup error without internal exception data."""
    return "The assistant could not start. Please contact the site administrator."
THINKING_LABEL = "🤔 Thinking…"


def chat_bubble_css():
    """Return scoped styles for user and assistant Streamlit chat bubbles."""
    return """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        overflow: hidden !important;
    }
    main#stMain {
        height: calc(100vh - 1rem) !important;
        overflow: hidden !important;
    }
    div[data-testid="stMainBlockContainer"] {
        height: 100%;
        overflow: hidden !important;
    }
    div.st-key-app_header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: var(--background-color);
        padding-top: 0.25rem;
    }
    div.st-key-chat_panel {
        height: max(220px, calc(100vh - 340px)) !important;
        max-height: 720px;
        min-height: 220px;
        flex: 0 0 auto;
        overflow-y: auto !important;
        overscroll-behavior: contain;
        scroll-behavior: smooth;
    }
    div.st-key-chat_panel > div {
        flex-shrink: 0;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
      [data-testid="stChatMessageContent"] {
        background: #2563eb;
        color: #ffffff;
        border-radius: 16px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
      [data-testid="stChatMessageContent"] {
        background: #111111;
        color: #ffffff;
        border-radius: 16px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stChatMessage"]:has([data-testid^="chatAvatarIcon-"])
      [data-testid="stChatMessageContent"] :is(p, li, strong, em, span, a) {
        color: #ffffff !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid^="chatAvatarIcon-"])
      [data-testid="stChatMessageContent"] a {
        text-decoration: underline;
    }
    </style>
    """


def auto_scroll_script():
    """Scroll the latest message using the chat panel's own scroll container."""
    return """
    <script>
    try {
      const page = window.parent.document;
      const chatPanel = page.querySelector('.st-key-chat_panel');
      if (chatPanel) {
        if (window.__buildmateChatScrollObserver) {
          window.__buildmateChatScrollObserver.disconnect();
        }
        if (window.__buildmateChatResizeObserver) {
          window.__buildmateChatResizeObserver.disconnect();
        }

        const scrollChatPanelToLatest = () => {
          const bottom = chatPanel.scrollHeight;
          chatPanel.scrollTop = bottom;
          chatPanel.scrollTo({ top: bottom, behavior: 'smooth' });
        };
        const scheduleScroll = () => page.defaultView.requestAnimationFrame(scrollChatPanelToLatest);
        const observer = new MutationObserver(scheduleScroll);
        observer.observe(chatPanel, { childList: true, subtree: true, characterData: true });
        window.__buildmateChatScrollObserver = observer;
        const resizeObserver = new ResizeObserver(scheduleScroll);
        resizeObserver.observe(chatPanel);
        if (chatPanel.firstElementChild) resizeObserver.observe(chatPanel.firstElementChild);
        window.__buildmateChatResizeObserver = resizeObserver;

        [0, 100, 300, 700, 1400].forEach(delay => window.setTimeout(scheduleScroll, delay));
        window.setTimeout(() => {
          scheduleScroll();
          observer.disconnect();
          resizeObserver.disconnect();
        }, 2500);
      }
    } catch (error) {
      // Keep chat usable when browser privacy settings block parent access.
    }
    </script>
    """


def select_prompt(typed_prompt=None, clicked_suggestions=()):
    """Prefer a clicked starter question, otherwise use text entered by the user."""
    return next((prompt for prompt in clicked_suggestions if prompt), typed_prompt)


def remaining_suggestions(suggestions, used_suggestions):
    """Return starter questions that have not already been selected."""
    return [question for question in suggestions if question not in used_suggestions]


def get_visible_suggestions(suggestions, used_suggestions, count=4):
    """Fill the visible starter-question slots with the next unused questions."""
    return remaining_suggestions(suggestions, used_suggestions)[:count]


def should_show_suggestions(history):
    """Only show starter questions after the customer has sent a message."""
    return any(message.get("role") == "user" for message in history)


def greeting_for_hour(hour):
    """Return a polite opening appropriate for India-local time."""
    if 5 <= hour < 12:
        salutation = "Good morning"
    elif 12 <= hour < 17:
        salutation = "Good afternoon"
    else:
        salutation = "Good evening"
    return f"{salutation}, sir/ma'am! How can I help you today?"


def submit_turn(assistant, prompt, history, lead, session_id, language=None):
    """Send a prompt to the assistant and return the updated chat state."""
    answer, updated_lead = assistant.respond(prompt, history, lead, session_id, language=language)
    updated_history = [
        *history,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, updated_lead, answer
