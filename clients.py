import os
import streamlit as st
from groq import Groq
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

def _safe_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default

GROQ_API_KEY = _safe_secret("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
GROQ_ACTIVE = bool(GROQ_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_ACTIVE else None

ANTHROPIC_API_KEY = _safe_secret("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
CLAUDE_MODEL = "claude-haiku-4-5"
anthropic_client = None
if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
    try:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        print("Anthropic init failed:", e)
        anthropic_client = None

CLAUDE_ACTIVE = anthropic_client is not None
