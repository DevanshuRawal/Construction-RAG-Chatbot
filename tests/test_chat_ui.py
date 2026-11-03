from src.chat_ui import (
    CHATBOT_NAME,
    THINKING_LABEL,
    auto_scroll_script,
    chat_bubble_css,
    get_visible_suggestions,
    greeting_for_hour,
    remaining_suggestions,
    select_prompt,
    should_show_suggestions,
    submit_turn,
)


def test_chatbot_has_friendly_product_name():
    assert CHATBOT_NAME == "BuildMate AI"


def test_thinking_indicator_has_clear_assistant_status():
    assert THINKING_LABEL == "🤔 Thinking…"


def test_chat_bubble_css_styles_user_and_assistant_roles():
    css = chat_bubble_css()

    assert 'chatAvatarIcon-user' in css
    assert 'background: #2563eb' in css
    assert 'chatAvatarIcon-assistant' in css
    assert 'background: #111111' in css
    assert 'color: #ffffff' in css
    assert 'st-key-chat_panel' in css
    assert 'overflow-y: auto' in css
    assert 'overscroll-behavior: contain' in css
    assert 'flex-shrink: 0' in css
    assert 'st-key-app_header' in css
    assert 'position: sticky' in css
    assert 'main#stMain' in css
    assert 'stAppViewContainer' in css
    assert 'overflow: hidden' in css


def test_auto_scroll_is_limited_to_the_chat_panel():
    script = auto_scroll_script()

    assert "querySelector('.st-key-chat_panel')" in script
    assert 'chatPanel.scrollTo' in script
    assert 'chatPanel.scrollHeight' in script
    assert 'chatPanel.scrollTop = bottom' in script
    assert 'scrollContainer' not in script
    assert 'requestAnimationFrame' in script
    assert 'MutationObserver' in script
    assert 'ResizeObserver' in script
    assert 'observer.observe(chatPanel' in script
    assert 'scrollIntoView' not in script
    assert "behavior: 'smooth'" in script


class FakeAssistant:
    def __init__(self):
        self.received = []

    def respond(self, prompt, history, lead, session_id, language="english"):
        self.received.append((prompt, history, lead, session_id, language))
        return f"Answer for: {prompt}", {**lead, "seen": True}


def test_clicked_suggestion_becomes_submitted_prompt():
    question = "What warranty do you give on modular kitchens?"

    assert select_prompt(typed_prompt=None, clicked_suggestions=[None, question, None]) == question


def test_submitted_suggestion_is_sent_to_assistant_and_added_to_history():
    assistant = FakeAssistant()
    history = [{"role": "assistant", "content": "Hi! How can I help?"}]
    lead = {}

    new_history, new_lead, answer = submit_turn(
        assistant, "What warranty do you give on modular kitchens?", history, lead, "session-1"
    )

    assert assistant.received[0][0] == "What warranty do you give on modular kitchens?"
    assert assistant.received[0][3] == "session-1"
    assert new_history[-2:] == [
        {"role": "user", "content": "What warranty do you give on modular kitchens?"},
        {"role": "assistant", "content": "Answer for: What warranty do you give on modular kitchens?"},
    ]
    assert new_lead == {"seen": True}
    assert answer.startswith("Answer for:")


def test_selected_reply_language_is_passed_to_assistant():
    assistant = FakeAssistant()

    submit_turn(assistant, "What is the warranty?", [], {}, "session-2", language="hindi")

    assert assistant.received[0][4] == "hindi"


def test_only_selected_suggestion_is_removed_from_remaining_questions():
    suggestions = ["Question one", "Question two", "Question three"]

    assert remaining_suggestions(suggestions, {"Question two"}) == ["Question one", "Question three"]


def test_suggestion_list_stays_at_four_and_replaces_selected_item():
    suggestions = [f"Question {index}" for index in range(1, 7)]

    assert get_visible_suggestions(suggestions, set()) == suggestions[:4]
    assert get_visible_suggestions(suggestions, {"Question 2"}) == [
        "Question 1", "Question 3", "Question 4", "Question 5"
    ]


def test_suggestions_wait_until_customer_sends_first_message():
    assert not should_show_suggestions([{"role": "assistant", "content": "Good morning!"}])
    assert should_show_suggestions([
        {"role": "assistant", "content": "Good morning!"},
        {"role": "user", "content": "Hello"},
    ])


def test_greeting_changes_with_india_time_of_day():
    assert greeting_for_hour(9) == "Good morning, sir/ma'am! How can I help you today?"
    assert greeting_for_hour(14) == "Good afternoon, sir/ma'am! How can I help you today?"
    assert greeting_for_hour(19) == "Good evening, sir/ma'am! How can I help you today?"


def test_unforced_language_is_left_for_assistant_to_detect_from_customer_message():
    assistant = FakeAssistant()

    submit_turn(assistant, "Mere ghar ka bathroom renovation kitna time lega?", [], {}, "session-3")

    assert assistant.received[0][4] is None
