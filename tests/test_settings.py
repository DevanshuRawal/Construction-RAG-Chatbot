from src.settings import load_streamlit_secrets


def test_streamlit_secrets_fill_missing_environment_values():
    env = {"COHERE_API_KEY": "local-key"}
    secrets = {
        "COHERE_API_KEY": "cloud-key",
        "GEMINI_API_KEY": "gemini-cloud-key",
        "GEMINI_MODEL": "gemini-3.8-flash",
        "LEADS_ENCRYPTION_KEY": "encrypted-lead-key",
        "LEADS_FILE": "leads.enc.jsonl",
    }

    load_streamlit_secrets(env, secrets)

    assert env["COHERE_API_KEY"] == "local-key"
    assert env["GEMINI_API_KEY"] == "gemini-cloud-key"
    assert env["GEMINI_MODEL"] == "gemini-3.8-flash"
    assert env["LEADS_ENCRYPTION_KEY"] == "encrypted-lead-key"
    assert env["LEADS_FILE"] == "leads.enc.jsonl"


def test_streamlit_secrets_ignore_empty_values():
    env = {}

    load_streamlit_secrets(env, {"COHERE_API_KEY": "", "HF_TOKEN": None})

    assert env == {}
