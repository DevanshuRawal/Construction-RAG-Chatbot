"""Runtime configuration helpers for local and Streamlit Cloud deployments."""

STREAMLIT_SECRET_KEYS = (
    "COHERE_API_KEY",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODEL",
    "HF_TOKEN",
    "HF_MODEL",
    "LEADS_ENCRYPTION_KEY",
    "LEADS_FILE",
    "LEADS_RETENTION_DAYS",
    "COHERE_CHAT_MODEL",
    "COHERE_FALLBACK_MODEL",
    "DATA_DIR",
    "CHROMA_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "DEBUG_LOGGING",
    "DEBUG_LOG_FILE",
    "MAX_CONCURRENT_LLM_REQUESTS",
)


def load_streamlit_secrets(env, secrets):
    """Copy configured Streamlit secrets into env without overriding local env."""
    for key in STREAMLIT_SECRET_KEYS:
        value = secrets.get(key)
        if key not in env and value not in (None, ""):
            env[key] = str(value)
