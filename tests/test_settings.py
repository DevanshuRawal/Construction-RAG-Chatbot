from src.settings import load_streamlit_secrets


def test_streamlit_secrets_fill_missing_environment_values():
    env = {"COHERE_API_KEY": "local-key"}
    secrets = {
        "COHERE_API_KEY": "cloud-key",
        "LEADS_ENCRYPTION_KEY": "encrypted-lead-key",
        "LEADS_FILE": "leads.enc.jsonl",
    }

    load_streamlit_secrets(env, secrets)

    assert env["COHERE_API_KEY"] == "local-key"
    assert env["LEADS_ENCRYPTION_KEY"] == "encrypted-lead-key"
    assert env["LEADS_FILE"] == "leads.enc.jsonl"


def test_streamlit_secrets_ignore_empty_values():
    env = {}

    load_streamlit_secrets(env, {"COHERE_API_KEY": "", "HF_TOKEN": None})

    assert env == {}
