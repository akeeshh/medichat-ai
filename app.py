import streamlit as st
from groq import Groq
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import os
import base64
import hashlib
from PIL import Image
import io
import re
import time
from datetime import datetime
from fpdf import FPDF

# Firebase for cross-session analytics (optional - fails gracefully if missing)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

st.set_page_config(
    page_title="MediChat - Your Health Assistant",
    page_icon="🏥",
    layout="centered"
)

# ── Firebase Initialization (cross-session analytics) ────────────────
@st.cache_resource
def init_firebase():
    """Initialize Firebase Admin SDK. Returns Firestore client or None if unavailable."""
    if not FIREBASE_AVAILABLE:
        return None
    try:
        firebase_config = st.secrets.get("firebase", {})
        if not firebase_config or not firebase_config.get("project_id"):
            return None
        if not firebase_admin._apps:
            cred = credentials.Certificate(dict(firebase_config))
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print("Firebase init failed:", e)
        return None

firestore_db = init_firebase()
FIREBASE_ACTIVE = firestore_db is not None

# ── Persistent Patient Profiles (email + PIN, Firestore-backed) ──────
PROFILE_SALT = st.secrets.get("PROFILE_SALT", os.environ.get("PROFILE_SALT", "medichat-default-change-me"))

def hash_email(email):
    return hashlib.sha256((email.lower().strip() + PROFILE_SALT).encode()).hexdigest()

def hash_pin(pin, email_hash):
    return hashlib.sha256((str(pin) + email_hash + PROFILE_SALT).encode()).hexdigest()

def get_profile(email_hash):
    if not FIREBASE_ACTIVE:
        return None
    try:
        doc = firestore_db.collection("medichat_profiles").document(email_hash).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print("Profile fetch failed:", e)
        return None

def create_profile(email, pin, name=""):
    if not FIREBASE_ACTIVE:
        return None
    eh = hash_email(email)
    profile = {
        "email_hash": eh,
        "pin_hash": hash_pin(pin, eh),
        "name": (name or "").strip()[:30],
        "patient_memory": {"symptoms": [], "conditions": [], "medications": []},
        "language": "English",
        "created_at": firestore.SERVER_TIMESTAMP,
        "last_visit": firestore.SERVER_TIMESTAMP,
        "visit_count": 1,
    }
    try:
        firestore_db.collection("medichat_profiles").document(eh).set(profile)
        profile["email_hash"] = eh
        return profile
    except Exception as e:
        print("Profile create failed:", e)
        return None

def authenticate_profile(email, pin):
    eh = hash_email(email)
    profile = get_profile(eh)
    if profile is None:
        return None, "not_found"
    if profile.get("pin_hash") != hash_pin(pin, eh):
        return None, "wrong_pin"
    try:
        firestore_db.collection("medichat_profiles").document(eh).update({
            "last_visit": firestore.SERVER_TIMESTAMP,
            "visit_count": firestore.Increment(1),
        })
    except Exception as e:
        print("Profile visit-count update failed:", e)
    profile["email_hash"] = eh
    return profile, "ok"

def persist_profile_state(email_hash, patient_memory=None, name=None, language=None, messages=None):
    if not FIREBASE_ACTIVE or not email_hash:
        return
    update = {"last_visit": firestore.SERVER_TIMESTAMP}
    if patient_memory is not None:
        update["patient_memory"] = patient_memory
    if name is not None:
        update["name"] = (name or "").strip()[:30]
    if language is not None:
        update["language"] = language
    if messages is not None:
        # Trim to last 60 messages, strip non-serialisable fields, cap content length
        trimmed = []
        for m in messages[-60:]:
            if not isinstance(m, dict):
                continue
            trimmed.append({
                "role": m.get("role", ""),
                "type": m.get("type", "text"),
                "content": (m.get("content", "") or "")[:4000],
                "sources": m.get("sources", []),
                "confidence": m.get("confidence", ""),
                "confidence_pct": m.get("confidence_pct", 0),
                "engine": m.get("engine", ""),
            })
        update["messages"] = trimmed
    try:
        firestore_db.collection("medichat_profiles").document(email_hash).update(update)
    except Exception as e:
        print("Profile state persist failed:", e)

def log_query_to_firestore(query_data):
    """Write anonymised query metadata to Firestore. Silent failure if Firebase unavailable."""
    if not FIREBASE_ACTIVE:
        return
    try:
        safe_data = {
            "query_word_count": len(query_data.get("query", "").split()),
            "confidence": query_data.get("confidence", "unknown"),
            "confidence_pct": query_data.get("confidence_pct", 0),
            "sources": query_data.get("sources", []),
            "response_time": query_data.get("response_time", 0),
            "language": query_data.get("language", "English"),
            "mode": query_data.get("mode", "free_chat"),
            "emergency_triggered": query_data.get("emergency_triggered", False),
            "drug_alerts": query_data.get("drug_alerts", 0),
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        firestore_db.collection("medichat_queries").add(safe_data)
    except Exception as e:
        print("Firestore write failed:", e)

def fetch_all_queries_from_firestore(limit=500):
    """Retrieve all anonymised query logs for admin dashboard."""
    if not FIREBASE_ACTIVE:
        return []
    try:
        docs = firestore_db.collection("medichat_queries").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        print("Firestore read failed:", e)
        return []

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Serif+Display:ital@0;1&display=swap');

    :root {
        --clinical-900: #0c2d48;
        --clinical-800: #144272;
        --clinical-700: #1a5b8a;
        --clinical-600: #2176ae;
        --clinical-500: #2a8fc5;
        --clinical-400: #4da8d6;
        --clinical-300: #7ec3e6;
        --clinical-200: #b0daf2;
        --clinical-100: #d6edf9;
        --clinical-50: #edf6fc;

        --neutral-900: #1a1d21;
        --neutral-800: #2d3238;
        --neutral-700: #4a5058;
        --neutral-600: #6b7280;
        --neutral-500: #9ca3af;
        --neutral-400: #cbd5e1;
        --neutral-300: #e2e8f0;
        --neutral-200: #f1f5f9;
        --neutral-100: #f8fafc;

        --accent-rose: #c4766a;
        --accent-amber: #d69e2e;
        --accent-lavender: #8b7aa8;
    }

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(ellipse 80% 60% at top left, rgba(42, 143, 197, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse 70% 50% at bottom right, rgba(139, 122, 168, 0.05) 0%, transparent 50%),
            linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        min-height: 100vh;
    }

    .main .block-container {
        padding: 1rem 1.5rem 2rem 1.5rem;
        max-width: 800px;
    }

    /* ── Header ──────────────────────────────────────────────────── */
    .header-card {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow:
            0 1px 2px rgba(12, 45, 72, 0.04),
            0 4px 16px rgba(12, 45, 72, 0.06);
        position: relative;
        overflow: hidden;
        border: 1px solid var(--clinical-100);
    }

    .header-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--clinical-500), var(--clinical-300), var(--clinical-500));
        background-size: 200% 100%;
        animation: shimmer 6s ease-in-out infinite;
    }

    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .header-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0;
    }

    .header-logo {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--clinical-500), var(--clinical-700));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        color: white;
        box-shadow: 0 3px 10px rgba(42, 143, 197, 0.25);
        flex-shrink: 0;
    }

    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--clinical-900);
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }

    .header-subtitle {
        color: var(--neutral-600);
        font-size: 0.78rem;
        margin: 0.15rem 0 0 0;
        font-weight: 400;
        line-height: 1.4;
    }

    /* ── Trust Strip ─────────────────────────────────────────────── */
    .trust-strip {
        display: flex;
        gap: 0.5rem;
        margin: 0.8rem 0 1rem 0;
        flex-wrap: wrap;
        justify-content: center;
    }

    .trust-pill {
        background: white;
        border: 1px solid var(--clinical-100);
        border-radius: 100px;
        padding: 0.35rem 0.85rem;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--clinical-700);
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        box-shadow: 0 1px 2px rgba(12, 45, 72, 0.03);
        transition: all 0.2s ease;
    }

    .trust-pill:hover {
        border-color: var(--clinical-300);
        transform: translateY(-1px);
    }

    .trust-pill-icon {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: var(--clinical-500);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.5rem;
        font-weight: 700;
    }

    .stats-row { display: flex; gap: 0.5rem; margin-bottom: 0.8rem; flex-wrap: wrap; justify-content: center; }
    .stat-pill { background: white; border: 1px solid var(--clinical-100); border-radius: 100px; padding: 0.35rem 0.85rem; font-size: 0.7rem; font-weight: 500; color: var(--clinical-700); }
    .stat-pill.green { color: var(--clinical-700); border-color: var(--clinical-300); background: var(--clinical-50); }
    .stat-pill.blue { color: #3a6b8f; border-color: #bed2e0; background: #eff5fa; }
    .stat-pill.purple { color: var(--accent-lavender); border-color: #d4c9e3; background: #f5f0fa; }
    .stat-pill.orange { color: var(--accent-amber); border-color: #e8cf9e; background: #fbf5e7; }

    /* ── Disclaimer ──────────────────────────────────────────────── */
    .disclaimer {
        display: none;
    }

    .disclaimer-mini {
        font-size: 0.68rem;
        color: var(--neutral-500);
        text-align: center;
        padding: 0.3rem 0;
        margin-top: 0.5rem;
        opacity: 0.8;
    }
    .disclaimer-mini-red {
        color: #a85c50;
    }

    /* ── Emergency Banner ────────────────────────────────────────── */
    .emergency-banner {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        color: white;
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.2);
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 6px 20px rgba(220, 38, 38, 0.2); }
        50% { box-shadow: 0 6px 30px rgba(220, 38, 38, 0.45); }
    }
    .emergency-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.5rem; }
    .emergency-text { font-size: 0.82rem; line-height: 1.5; margin-bottom: 0.5rem; opacity: 0.95; }
    .emergency-number {
        background: white; color: #991b1b;
        padding: 0.5rem 1.1rem; border-radius: 10px;
        font-size: 1.1rem; font-weight: 700;
        display: inline-block; margin-top: 0.15rem;
        letter-spacing: 0.04em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* ── Mode Buttons ────────────────────────────────────────────── */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: 1.5px solid var(--clinical-300) !important;
        background: white !important;
        color: #1a1d21 !important;
        padding: 0 1rem !important;
        height: 44px !important;
        min-height: 44px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(12, 45, 72, 0.08) !important;
        font-size: 0.85rem !important;
    }
    .stButton > button:hover {
        border-color: var(--clinical-500) !important;
        background: var(--clinical-50) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(42, 143, 197, 0.15) !important;
    }

    /* ── Primary Send button (chat form only — uses type="primary") ── */
    .stForm [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
    .stForm [data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, var(--clinical-600), var(--clinical-800)) !important;
        color: white !important;
        border: 1px solid var(--clinical-700) !important;
        font-weight: 600 !important;
        height: 44px !important;
        min-height: 44px !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 12px rgba(12, 45, 72, 0.2) !important;
    }
    .stForm [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover,
    .stForm [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--clinical-800), var(--clinical-900)) !important;
        border-color: var(--clinical-900) !important;
        box-shadow: 0 6px 18px rgba(12, 45, 72, 0.3) !important;
        transform: translateY(-1px);
    }

    /* Secondary form buttons (Save / Skip / Next / Cancel) — visible dark text */
    .stForm [data-testid="stFormSubmitButton"] > button[kind="secondaryFormSubmit"],
    .stForm [data-testid="stFormSubmitButton"] > button[kind="secondary"] {
        background: white !important;
        color: var(--clinical-900) !important;
        border: 1px solid var(--clinical-300) !important;
        font-weight: 600 !important;
        height: 44px !important;
        min-height: 44px !important;
        border-radius: 14px !important;
    }
    .stForm [data-testid="stFormSubmitButton"] > button[kind="secondaryFormSubmit"]:hover,
    .stForm [data-testid="stFormSubmitButton"] > button[kind="secondary"]:hover {
        background: var(--clinical-50) !important;
        border-color: var(--clinical-500) !important;
        color: var(--clinical-900) !important;
    }


    /* Match input height to buttons */
    .stForm .stTextInput > div > div > input {
        height: 44px !important;
        padding: 0 1.1rem !important;
    }

    /* ── Welcome Card ────────────────────────────────────────────── */
    .welcome-card {
        background: white;
        border-radius: 18px;
        padding: 2rem 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(12, 45, 72, 0.06);
        margin: 0.5rem 0 1rem 0;
        border: 1px solid var(--clinical-100);
        animation: fadeInUp 0.5s ease-out;
    }

    .hero-wrap {
        background: linear-gradient(135deg, #f0f7fc 0%, #e3f0f9 100%);
        border-radius: 18px;
        padding: 1.4rem 1.6rem 1.3rem 1.6rem;
        margin: 0 0 0.8rem 0;
        position: relative;
        overflow: hidden;
        border: 1px solid var(--clinical-100);
        animation: fadeInUp 0.6s ease-out;
    }
    .hero-wrap::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(42, 143, 197, 0.08), transparent 70%);
        border-radius: 50%;
        z-index: 0;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: white;
        border: 1px solid var(--clinical-100);
        color: var(--clinical-700);
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.25rem 0.7rem;
        border-radius: 100px;
        margin-bottom: 0.6rem;
        position: relative;
        z-index: 1;
    }
    .hero-eyebrow::before {
        content: "●";
        color: #22c55e;
        font-size: 0.45rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--clinical-900);
        line-height: 1.2;
        letter-spacing: -0.01em;
        margin-bottom: 0.4rem;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        color: var(--neutral-600);
        font-size: 0.82rem;
        line-height: 1.5;
        max-width: 560px;
        margin-bottom: 0.8rem;
        position: relative;
        z-index: 1;
    }
    .hero-trust-row {
        display: flex;
        gap: 0.3rem;
        flex-wrap: wrap;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
    }
    .trust-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: white;
        border: 1px solid var(--clinical-100);
        color: var(--clinical-700);
        font-size: 0.68rem;
        font-weight: 500;
        padding: 0.3rem 0.65rem;
        border-radius: 100px;
        box-shadow: 0 1px 3px rgba(12, 45, 72, 0.04);
    }
    .trust-icon {
        width: 12px;
        height: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .welcome-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--clinical-900);
        margin: 0.6rem 0 0.5rem 0;
        letter-spacing: -0.01em;
    }
    .welcome-text {
        color: var(--neutral-600);
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    .chip-row {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 0.8rem;
    }
    .chip {
        background: var(--clinical-50);
        border: 1px solid var(--clinical-100);
        color: var(--clinical-700);
        padding: 0.35rem 0.85rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .chip:hover {
        background: var(--clinical-100);
        border-color: var(--clinical-300);
        transform: translateY(-1px);
    }

    /* ── Memory Card ─────────────────────────────────────────────── */
    .memory-card {
        background: linear-gradient(135deg, var(--clinical-50), #e3f0f9);
        border: 1px solid var(--clinical-100);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.78rem;
        color: var(--clinical-700);
        animation: fadeIn 0.4s ease;
    }
    .memory-title { font-weight: 600; margin-bottom: 0.3rem; font-size: 0.8rem; color: var(--clinical-900); display: flex; align-items: center; gap: 0.35rem; }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* ── Chat Messages ───────────────────────────────────────────── */
    .bot-label {
        font-size: 0.65rem;
        color: var(--neutral-500);
        font-weight: 600;
        margin-left: 50px;
        margin-bottom: 0.25rem;
        margin-top: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .bot-wrap, .user-wrap {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        margin-bottom: 0.7rem;
        animation: messageSlideIn 0.35s ease-out;
    }

    .user-wrap {
        justify-content: flex-end;
    }

    @keyframes messageSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .av {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.9rem;
        flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(12, 45, 72, 0.1);
    }

    .av-bot {
        background: linear-gradient(135deg, var(--clinical-500), var(--clinical-700));
        color: white;
    }

    .av-user {
        background: linear-gradient(135deg, var(--clinical-100), var(--clinical-200));
        color: var(--clinical-900);
    }

    .bot-bubble {
        background: white;
        color: var(--neutral-800);
        padding: 0.9rem 1.15rem;
        border-radius: 4px 16px 16px 16px;
        max-width: 85%;
        font-size: 0.9rem;
        line-height: 1.6;
        box-shadow: 0 1px 2px rgba(12, 45, 72, 0.04), 0 4px 16px rgba(12, 45, 72, 0.04);
        border: 1px solid var(--clinical-100);
        position: relative;
    }

    .bot-bubble::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--clinical-500), var(--clinical-300));
        border-radius: 4px 0 0 0;
    }

    .bot-bubble .md-h {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--clinical-900);
        margin: 0.5rem 0 0.35rem 0;
        line-height: 1.25;
        letter-spacing: -0.01em;
    }
    .bot-bubble h3.md-h { font-size: 1.1rem; }
    .bot-bubble h4.md-h { font-size: 1rem; }
    .bot-bubble h5.md-h { font-size: 0.95rem; font-weight: 600; color: var(--clinical-700); }
    .bot-bubble h6.md-h { font-size: 0.9rem; font-weight: 600; color: var(--clinical-700); }
    .bot-bubble .md-h:first-child { margin-top: 0; }
    .bot-bubble .md-p { margin: 0.45rem 0; }
    .bot-bubble .md-p:first-child { margin-top: 0; }
    .bot-bubble .md-p:last-child { margin-bottom: 0; }
    .bot-bubble .md-ul, .bot-bubble .md-ol {
        margin: 0.45rem 0 0.5rem 0;
        padding-left: 1.3rem;
    }
    .bot-bubble .md-ul li, .bot-bubble .md-ol li {
        margin: 0.2rem 0;
        line-height: 1.5;
    }
    .bot-bubble .md-ul li::marker { color: var(--clinical-500); }
    .bot-bubble .md-ol li::marker { color: var(--clinical-500); font-weight: 600; }
    .bot-bubble strong { color: var(--clinical-900); font-weight: 600; }
    .bot-bubble em { font-style: italic; color: var(--clinical-700); }
    .bot-bubble .md-hr { border: none; border-top: 1px solid var(--clinical-100); margin: 0.7rem 0; }
    .bot-bubble .md-code {
        background: var(--clinical-50);
        color: var(--clinical-700);
        padding: 0.1rem 0.35rem;
        border-radius: 5px;
        font-family: 'SF Mono', Monaco, Consolas, monospace;
        font-size: 0.85em;
    }

    .user-bubble {
        background: linear-gradient(135deg, var(--clinical-600), var(--clinical-800));
        color: white;
        padding: 0.8rem 1.1rem;
        border-radius: 16px 4px 16px 16px;
        max-width: 80%;
        font-size: 0.9rem;
        line-height: 1.5;
        box-shadow: 0 4px 12px rgba(33, 118, 174, 0.2);
    }

    /* ── Thinking Indicator ──────────────────────────────────────── */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.7rem 0.9rem;
        background: white;
        border-radius: 4px 16px 16px 16px;
        border-left: 3px solid var(--clinical-500);
        max-width: 160px;
        box-shadow: 0 1px 2px rgba(12, 45, 72, 0.04);
    }
    .thinking-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--clinical-500);
        animation: bounce 1.4s infinite ease-in-out;
    }
    .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }
    .thinking-label {
        font-size: 0.75rem;
        color: var(--neutral-500);
        margin-left: 0.25rem;
        font-style: italic;
    }

    /* ── Streaming Cursor ────────────────────────────────────────── */
    .stream-cursor {
        display: inline-block;
        width: 2px;
        height: 1em;
        background: var(--clinical-500);
        margin-left: 2px;
        vertical-align: text-bottom;
        animation: blink 1s step-end infinite;
    }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }

    /* ── Source & Confidence Rows ────────────────────────────────── */
    .source-row {
        margin-left: 50px;
        margin-bottom: 0.25rem;
        font-size: 0.7rem;
        color: var(--neutral-500);
    }

    .engine-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.62rem;
        font-weight: 600;
        padding: 0.12rem 0.5rem;
        border-radius: 100px;
        letter-spacing: 0.02em;
        margin-right: 0.35rem;
    }
    .engine-claude {
        background: linear-gradient(135deg, #fef2e8, #fdeede);
        color: #a8521a;
        border: 1px solid #f5c4a1;
    }
    .engine-groq {
        background: var(--clinical-50);
        color: var(--clinical-700);
        border: 1px solid var(--clinical-300);
    }
    .engine-vision {
        background: #f3e8ff;
        color: #6b21a8;
        border: 1px solid #d8b4fe;
    }
    .engine-badge::before { content: "●"; font-size: 0.45rem; }
    .source-tag {
        display: inline-block;
        background: var(--clinical-50);
        border: 1px solid var(--clinical-100);
        color: var(--clinical-700);
        font-size: 0.65rem;
        font-weight: 500;
        padding: 0.15rem 0.6rem;
        border-radius: 100px;
        margin-right: 0.25rem;
        margin-top: 0.25rem;
    }
    .confidence-row {
        margin-left: 50px;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.65rem;
    }
    .confidence-pill {
        padding: 0.12rem 0.65rem;
        border-radius: 100px;
        font-weight: 600;
        letter-spacing: 0.03em;
        font-size: 0.63rem;
    }
    .conf-high { background: var(--clinical-50); color: var(--clinical-700); border: 1px solid var(--clinical-300); }
    .conf-medium { background: #fbf5e7; color: #7a5d1a; border: 1px solid #e8cf9e; }
    .conf-low { background: #fdf2f1; color: #8f3f34; border: 1px solid #e3bfb8; }
    .confidence-bar {
        display: inline-block;
        width: 75px;
        height: 5px;
        background: var(--clinical-100);
        border-radius: 100px;
        overflow: hidden;
    }
    .confidence-fill {
        display: block;
        height: 100%;
        border-radius: 100px;
    }

    /* ── Input ───────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 1.5px solid var(--clinical-100) !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.9rem !important;
        background: white !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--clinical-500) !important;
        box-shadow: 0 0 0 3px rgba(42, 143, 197, 0.15) !important;
    }

    /* ── Attach Uploader: styled to look like a clean Attach button ── */
    [data-testid="stFileUploader"] > label {
        display: none !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
        border: 1px solid var(--clinical-200) !important;
        border-radius: 14px !important;
        background: white !important;
        box-shadow: 0 1px 2px rgba(12, 45, 72, 0.04) !important;
        transition: all 0.2s ease !important;
        min-height: 44px !important;
        height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--clinical-400) !important;
        background: var(--clinical-50) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(42, 143, 197, 0.12) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: 0 1rem !important;
        min-height: 44px !important;
        height: 44px !important;
        background: transparent !important;
        border: none !important;
        border-radius: 14px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        height: 100% !important;
    }
    /* Hide Streamlit's original dropzone children */
    [data-testid="stFileUploaderDropzoneInstructions"] > * {
        visibility: hidden !important;
        position: absolute !important;
        pointer-events: none !important;
    }
    /* Inject our own visible label */
    [data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "📎  Attach image or PDF";
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--clinical-900) !important;
        letter-spacing: 0.01em;
        pointer-events: none;
    }
    [data-testid="stFileUploaderDropzone"] button {
        display: none !important;
    }
    /* Show filename neatly once a file is selected */
    [data-testid="stFileUploaderFile"] {
        padding: 0.5rem 0.8rem !important;
        background: var(--clinical-50) !important;
        border-radius: 12px !important;
        margin-top: 0.4rem !important;
    }


    /* ── Image Tag ───────────────────────────────────────────────── */
    .image-tag {
        display: inline-block;
        background: #f5f0fa;
        color: var(--accent-lavender);
        border: 1px solid #d4c9e3;
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }

    /* ── Name Welcome ────────────────────────────────────────────── */
    .name-welcome {
        background: linear-gradient(135deg, var(--clinical-50), #e3f0f9);
        border: 1px solid var(--clinical-100);
        border-radius: 12px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.8rem;
        animation: fadeInUp 0.4s ease;
    }
    .name-welcome-text {
        font-size: 0.88rem;
        color: var(--clinical-900);
        font-weight: 400;
    }

    /* ── Suggested Follow-ups ────────────────────────────────────── */
    .suggestion-row {
        margin-left: 50px;
        margin-bottom: 0.8rem;
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
    }
    .suggestion-label {
        font-size: 0.65rem;
        color: var(--neutral-500);
        margin-left: 50px;
        margin-bottom: 0.35rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Sidebar Styling ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 1px solid var(--clinical-100);
    }
    [data-testid="stSidebar"] .stMarkdown { color: var(--clinical-900); }
    .sb-title { font-size: 0.65rem; font-weight: 700; color: var(--neutral-500); text-transform: uppercase; letter-spacing: 0.12em; margin: 0.7rem 0 0.45rem 0; }
    .sb-stat-card { background: var(--clinical-50); border: 1px solid var(--clinical-100); border-radius: 10px; padding: 0.55rem 0.75rem; margin-bottom: 0.35rem; }
    .sb-stat-num { font-size: 1.3rem; font-weight: 700; color: var(--clinical-700) !important; line-height: 1; font-family: 'Inter', sans-serif; }
    .sb-stat-label { font-size: 0.62rem; color: var(--neutral-500) !important; font-weight: 500; margin-top: 0.1rem; }
    .sb-feature { display: flex; align-items: center; gap: 0.5rem; background: var(--clinical-50); border: 1px solid var(--clinical-100); border-radius: 8px; padding: 0.4rem 0.65rem; margin-bottom: 0.3rem; }
    .sb-feature-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
    .sb-feature-name { font-size: 0.72rem; font-weight: 500; color: var(--clinical-900) !important; }
    .sb-feature-status { font-size: 0.6rem; color: var(--clinical-500) !important; margin-left: auto; font-weight: 600; }
    .sb-tip { font-size: 0.7rem; color: var(--neutral-600) !important; padding: 0.25rem 0; border-bottom: 1px solid var(--clinical-50); line-height: 1.5; }
    .sb-memory-item { font-size: 0.68rem; color: var(--clinical-700) !important; padding: 0.2rem 0; border-bottom: 1px solid var(--clinical-50); }
    .sb-footer { font-size: 0.62rem; color: var(--neutral-500) !important; text-align: center; padding-top: 0.8rem; border-top: 1px solid var(--clinical-50); line-height: 1.6; }

    /* ── Symptom Assessment Card ─────────────────────────────────── */
    .assessment-card {
        background: white;
        border-radius: 18px;
        padding: 1.4rem 1.6rem 1.5rem 1.6rem;
        margin-bottom: 1.2rem;
        border: 1px solid var(--clinical-100);
        box-shadow: 0 1px 3px rgba(12, 45, 72, 0.04), 0 8px 28px rgba(12, 45, 72, 0.05);
        animation: fadeInUp 0.4s ease;
    }
    .assessment-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.45rem;
        font-weight: 400;
        color: var(--clinical-900);
        letter-spacing: -0.01em;
        margin-bottom: 0.35rem;
        line-height: 1.2;
    }
    .assessment-subtitle {
        color: var(--neutral-600, #6b6660);
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }
    .progress-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--clinical-700);
        margin-bottom: 0.45rem;
        letter-spacing: 0.02em;
    }
    .progress-label span:first-child {
        color: var(--clinical-900);
    }
    .progress-label span:last-child {
        color: var(--clinical-600);
        font-weight: 700;
    }
    .progress-bar-wrap {
        width: 100%;
        height: 8px;
        background: var(--clinical-50);
        border: 1px solid var(--clinical-100);
        border-radius: 100px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--clinical-500), var(--clinical-700));
        border-radius: 100px;
        transition: width 0.4s ease;
    }
    .question-bubble {
        background: var(--clinical-50);
        border: 1px solid var(--clinical-100);
        border-left: 4px solid var(--clinical-600);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0 1rem 0;
        font-size: 1rem;
        font-weight: 500;
        color: var(--clinical-900);
        line-height: 1.5;
    }

    /* ── Hide Streamlit Branding ─────────────────────────────────── */
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }

    .bot-bubble a[class*="anchor"],
    .bot-bubble svg[class*="anchor"],
    .bot-bubble a[href^="#"]:not([href*="://"]) { display: none !important; }
    [data-testid="stMarkdownContainer"] a.heading-anchor-icon,
    [data-testid="stMarkdownContainer"] a[href^="#"] svg { display: none !important; }

    [data-testid="InputInstructions"] { display: none !important; }
    .stForm [data-testid="stFormSubmitButton"] + small { display: none !important; }
    .stForm small { display: none !important; }

    [data-testid="stFileUploader"] [data-testid="stAlert"] {
        margin-top: 0.5rem !important;
        position: relative !important;
        z-index: 5 !important;
        display: block !important;
    }

</style>

""", unsafe_allow_html=True)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("API key not found.")
    st.stop()
groq_client = Groq(api_key=GROQ_API_KEY)

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
CLAUDE_MODEL = "claude-haiku-4-5"
anthropic_client = None
if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
    try:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        print("Anthropic init failed:", e)
        anthropic_client = None

CLAUDE_ACTIVE = anthropic_client is not None

# ── Language Config ───────────────────────────────────────────────────
LANGUAGES = {
    "English": {
        "flag": "🇦🇺",
        "greeting": "Hello! How can I help you today?",
        "welcome_text": "I am MediChat, your friendly AI health assistant.<br>I remember everything you tell me during our conversation.<br><br>Or switch to Symptom Check for a guided assessment!",
        "placeholder": "Type your health question here...",
        "send_btn": "Send to MediChat",
        "clear_btn": "Clear",
        "upload_label": "Upload a medical image (optional)",
        "question_label": "Your question",
        "helpful": "Was this conversation helpful?",
        "yes": "Yes",
        "no": "No",
        "thanks_helpful": "Thank you! Glad MediChat was helpful.",
        "thanks_not": "Thank you for your feedback. We will keep improving!",
        "download_chat": "Download your conversation:",
        "download_chat_btn": "Download Chat as PDF",
        "download_assess_btn": "Download Assessment Report as PDF",
        "symptom_title": "Symptom Assessment",
        "symptom_subtitle": "Answer a few quick questions and MediChat will generate a personalised health assessment.",
        "quick_select": "Quick select:",
        "next": "Next",
        "cancel": "Cancel",
        "answers_so_far": "Your answers so far",
        "new_assessment": "Start New Assessment",
        "switch_chat": "Switch to Free Chat",
        "report_title": "MediChat Assessment Report",
        "symptoms_reported": "Symptoms Reported",
        "possible_conditions": "Possible Conditions",
        "what_to_do": "What To Do Next",
        "summary": "MediChat Summary",
        "disclaimer_short": "This assessment is for information only and is NOT a medical diagnosis. Please consult a qualified healthcare professional.",
        "free_chat": "Free Chat",
        "symptom_check": "Symptom Check",
        "lang_instruction": "IMPORTANT: You MUST respond entirely in English.",
    },
    "Tamil": {
        "flag": "🇱🇰",
        "greeting": "வணக்கம்! இன்று நான் உங்களுக்கு எப்படி உதவலாம்?",
        "welcome_text": "நான் MediChat — உங்கள் நட்பான AI சுகாதார உதவியாளர்.<br>உரையாடலில் நீங்கள் சொல்வதை நான் நினைவில் வைத்திருப்பேன்.<br><br>வழிகாட்டப்பட்ட மதிப்பீட்டிற்கு அறிகுறி சரிபார்ப்புக்கு மாறலாம்!",
        "placeholder": "உங்கள் உடல்நல கேள்வியை இங்கே தட்டச்சு செய்யுங்கள்...",
        "send_btn": "MediChat க்கு அனுப்பவும்",
        "clear_btn": "அழிக்கவும்",
        "upload_label": "மருத்துவ படத்தை பதிவேற்றவும் (விரும்பினால்)",
        "question_label": "உங்கள் கேள்வி",
        "helpful": "இந்த உரையாடல் உதவியாக இருந்ததா?",
        "yes": "ஆம்",
        "no": "இல்லை",
        "thanks_helpful": "நன்றி! MediChat உதவியாக இருந்தது மகிழ்ச்சி.",
        "thanks_not": "உங்கள் கருத்துக்கு நன்றி. நாங்கள் தொடர்ந்து மேம்படுவோம்!",
        "download_chat": "உரையாடலை பதிவிறக்கவும்:",
        "download_chat_btn": "அரட்டையை PDF ஆக பதிவிறக்கவும்",
        "download_assess_btn": "மதிப்பீட்டு அறிக்கையை PDF ஆக பதிவிறக்கவும்",
        "symptom_title": "அறிகுறி மதிப்பீடு",
        "symptom_subtitle": "சில விரைவான கேள்விகளுக்கு பதிலளிக்கவும், MediChat உங்களுக்கு தனிப்பயன் சுகாதார மதிப்பீட்டை உருவாக்கும்.",
        "quick_select": "விரைவு தேர்வு:",
        "next": "அடுத்து",
        "cancel": "ரத்து செய்",
        "answers_so_far": "இதுவரை உங்கள் பதில்கள்",
        "new_assessment": "புதிய மதிப்பீட்டை தொடங்கவும்",
        "switch_chat": "இலவச அரட்டைக்கு மாறவும்",
        "report_title": "MediChat மதிப்பீட்டு அறிக்கை",
        "symptoms_reported": "தெரிவிக்கப்பட்ட அறிகுறிகள்",
        "possible_conditions": "சாத்தியமான நிலைமைகள்",
        "what_to_do": "அடுத்து என்ன செய்வது",
        "summary": "MediChat சுருக்கம்",
        "disclaimer_short": "இந்த மதிப்பீடு தகவல் நோக்கங்களுக்காக மட்டுமே. தயவுசெய்து தகுதிவாய்ந்த மருத்துவரை அணுகவும்.",
        "free_chat": "இலவச அரட்டை",
        "symptom_check": "அறிகுறி சரிபார்ப்பு",
        "lang_instruction": "IMPORTANT: You MUST respond entirely in Tamil (தமிழ்). All your responses must be in Tamil language.",
    },
    "Sinhala": {
        "flag": "🇱🇰",
        "greeting": "ආයුබෝවන්! අද මට ඔබට කෙසේ උදව් කළ හැකිද?",
        "welcome_text": "මම MediChat — ඔබේ මිත්‍රශීලී AI සෞඛ්‍ය සහායකයා.<br>ඔබ කියන සෑම දෙයක්ම මම මතක තබා ගනිමි.<br><br>මඟ පෙන්වූ තක්සේරු කිරීම සඳහා රෝග ලක්ෂණ පරීක්ෂාවට මාරු වන්න!",
        "placeholder": "ඔබේ සෞඛ්‍ය ප්‍රශ්නය මෙහි ටයිප් කරන්න...",
        "send_btn": "MediChat වෙත යවන්න",
        "clear_btn": "හිස් කරන්න",
        "upload_label": "වෛද්‍ය රූපයක් උඩුගත කරන්න (විකල්ප)",
        "question_label": "ඔබේ ප්‍රශ්නය",
        "helpful": "මෙම සංවාදය ප්‍රයෝජනවත් වූවාද?",
        "yes": "ඔව්",
        "no": "නැහැ",
        "thanks_helpful": "ස්තූතියි! MediChat ප්‍රයෝජනවත් වූ බව සතුටක්.",
        "thanks_not": "ඔබේ ප්‍රතිචාරයට ස්තූතියි. අපි දිගටම වැඩිදියුණු කරන්නෙමු!",
        "download_chat": "ඔබේ සංවාදය බාගත කරන්න:",
        "download_chat_btn": "Chat PDF ලෙස බාගත කරන්න",
        "download_assess_btn": "තක්සේරු වාර්තාව PDF ලෙස බාගත කරන්න",
        "symptom_title": "රෝග ලක්ෂණ තක්සේරු කිරීම",
        "symptom_subtitle": "ප්‍රශ්න කිහිපයකට පිළිතුරු දෙන්න, MediChat ඔබට පෞද්ගලික සෞඛ්‍ය තක්සේරුවක් ජනනය කරනු ඇත.",
        "quick_select": "ඉක්මන් තේරීම:",
        "next": "ඊළඟ",
        "cancel": "අවලංගු කරන්න",
        "answers_so_far": "මෙතෙක් ඔබේ පිළිතුරු",
        "new_assessment": "නව තක්සේරු කිරීමක් ආරම්භ කරන්න",
        "switch_chat": "නිදහස් Chat වෙත මාරු වන්න",
        "report_title": "MediChat තක්සේරු වාර්තාව",
        "symptoms_reported": "වාර්තා කළ රෝග ලක්ෂණ",
        "possible_conditions": "හැකි තත්ත්වයන්",
        "what_to_do": "ඊළඟට කළ යුතු දේ",
        "summary": "MediChat සාරාංශය",
        "disclaimer_short": "මෙම තක්සේරු කිරීම තොරතුරු පමණක් වේ. සුදුසුකම් ලත් වෛද්‍යවරයෙකු හමුවෙන්න.",
        "free_chat": "නිදහස් Chat",
        "symptom_check": "රෝග ලක්ෂණ පරීක්ෂාව",
        "lang_instruction": "IMPORTANT: You MUST respond entirely in Sinhala (සිංහල). All your responses must be in Sinhala language.",
    },
    "Hindi": {
        "flag": "🇮🇳",
        "greeting": "नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूं?",
        "welcome_text": "मैं MediChat हूं — आपका मित्रवत AI स्वास्थ्य सहायक.<br>आप जो कुछ भी बताते हैं मैं याद रखता हूं.<br><br>निर्देशित मूल्यांकन के लिए लक्षण जांच पर स्विच करें!",
        "placeholder": "अपना स्वास्थ्य प्रश्न यहाँ टाइप करें...",
        "send_btn": "MediChat को भेजें",
        "clear_btn": "साफ करें",
        "upload_label": "चिकित्सा छवि अपलोड करें (वैकल्पिक)",
        "question_label": "आपका प्रश्न",
        "helpful": "क्या यह बातचीत मददगार थी?",
        "yes": "हाँ",
        "no": "नहीं",
        "thanks_helpful": "धन्यवाद! खुशी है कि MediChat मददगार रहा।",
        "thanks_not": "आपकी प्रतिक्रिया के लिए धन्यवाद। हम लगातार सुधार करते रहेंगे!",
        "download_chat": "अपनी बातचीत डाउनलोड करें:",
        "download_chat_btn": "चैट PDF के रूप में डाउनलोड करें",
        "download_assess_btn": "मूल्यांकन रिपोर्ट PDF डाउनलोड करें",
        "symptom_title": "लक्षण मूल्यांकन",
        "symptom_subtitle": "कुछ त्वरित प्रश्नों का उत्तर दें और MediChat आपका व्यक्तिगत स्वास्थ्य मूल्यांकन तैयार करेगा।",
        "quick_select": "त्वरित चयन:",
        "next": "अगला",
        "cancel": "रद्द करें",
        "answers_so_far": "अब तक के आपके उत्तर",
        "new_assessment": "नया मूल्यांकन शुरू करें",
        "switch_chat": "फ्री चैट पर स्विच करें",
        "report_title": "MediChat मूल्यांकन रिपोर्ट",
        "symptoms_reported": "रिपोर्ट किए गए लक्षण",
        "possible_conditions": "संभावित स्थितियां",
        "what_to_do": "आगे क्या करें",
        "summary": "MediChat सारांश",
        "disclaimer_short": "यह मूल्यांकन केवल सूचना के लिए है। कृपया योग्य चिकित्सक से परामर्श लें।",
        "free_chat": "फ्री चैट",
        "symptom_check": "लक्षण जांच",
        "lang_instruction": "IMPORTANT: You MUST respond entirely in Hindi (हिन्दी). All your responses must be in Hindi language.",
    },
    "Malayalam": {
        "flag": "🇮🇳",
        "greeting": "നമസ്കാരം! ഇന്ന് ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",
        "welcome_text": "ഞാൻ MediChat — നിങ്ങളുടെ സൗഹൃദ AI ആരോഗ്യ സഹായി.<br>നിങ്ങൾ പറയുന്നതെല്ലാം ഞാൻ ഓർത്തിരിക്കും.<br><br>ഗൈഡഡ് അസസ്മെൻ്റിനായി സിംപ്റ്റം ചെക്കിലേക്ക് മാറുക!",
        "placeholder": "നിങ്ങളുടെ ആരോഗ്യ ചോദ്യം ഇവിടെ ടൈപ്പ് ചെയ്യുക...",
        "send_btn": "MediChat-ലേക്ക് അയയ്ക്കുക",
        "clear_btn": "മായ്ക്കുക",
        "upload_label": "മെഡിക്കൽ ചിത്രം അപ്ലോഡ് ചെയ്യുക (ഐച്ഛികം)",
        "question_label": "നിങ്ങളുടെ ചോദ്യം",
        "helpful": "ഈ സംഭാഷണം സഹായകരമായിരുന്നോ?",
        "yes": "അതെ",
        "no": "ഇല്ല",
        "thanks_helpful": "നന്ദി! MediChat സഹായകരമായതിൽ സന്തോഷം.",
        "thanks_not": "നിങ്ങളുടെ ഫീഡ്‌ബാക്കിന് നന്ദി. ഞങ്ങൾ മെച്ചപ്പെടുത്തുന്നത് തുടരും!",
        "download_chat": "നിങ്ങളുടെ സംഭാഷണം ഡൗൺലോഡ് ചെയ്യുക:",
        "download_chat_btn": "ചാറ്റ് PDF ആയി ഡൗൺലോഡ് ചെയ്യുക",
        "download_assess_btn": "അസസ്മെൻ്റ് റിപ്പോർട്ട് PDF ഡൗൺലോഡ് ചെയ്യുക",
        "symptom_title": "ലക്ഷണ വിലയിരുത്തൽ",
        "symptom_subtitle": "ചില ചോദ്യങ്ങൾക്ക് ഉത്തരം നൽകൂ, MediChat നിങ്ങൾക്ക് വ്യക്തിഗത ആരോഗ്യ വിലയിരുത്തൽ തയ്യാറാക്കും.",
        "quick_select": "വേഗ തിരഞ്ഞെടുക്കൽ:",
        "next": "അടുത്തത്",
        "cancel": "റദ്ദാക്കുക",
        "answers_so_far": "ഇതുവരെ നിങ്ങളുടെ ഉത്തരങ്ങൾ",
        "new_assessment": "പുതിയ വിലയിരുത്തൽ ആരംഭിക്കുക",
        "switch_chat": "ഫ്രീ ചാറ്റിലേക്ക് മാറുക",
        "report_title": "MediChat വിലയിരുത്തൽ റിപ്പോർട്ട്",
        "symptoms_reported": "റിപ്പോർട്ട് ചെയ്ത ലക്ഷണങ്ങൾ",
        "possible_conditions": "സാധ്യമായ അവസ്ഥകൾ",
        "what_to_do": "അടുത്തതായി എന്ത് ചെയ്യണം",
        "summary": "MediChat സംഗ്രഹം",
        "disclaimer_short": "ഈ വിലയിരുത്തൽ വിവരങ്ങൾക്ക് മാത്രമുള്ളതാണ്. യോഗ്യതയുള്ള ഡോക്ടറെ സമീപിക്കുക.",
        "free_chat": "ഫ്രീ ചാറ്റ്",
        "symptom_check": "ലക്ഷണ പരിശോധന",
        "lang_instruction": "IMPORTANT: You MUST respond entirely in Malayalam (മലയാളം). All your responses must be in Malayalam language.",
    },
}

@st.cache_resource
def load_rag_system():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    pubmed_docs = []
    dialog_docs = []

    try:
        pubmed = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:500]")
        pubmed_docs = ["[PubMed Research]\nQuestion: " + i["question"] + "\nAnswer: " + i["long_answer"] for i in pubmed]
    except Exception as e:
        print("PubMedQA load failed:", e)

    dialog_sources = [
        ("BinKhoaLe1812/MedDialog-EN-100k", "train[:500]", "input", "output"),
        ("shibing624/medical", "train[:500]", "instruction", "output"),
    ]
    for ds_name, ds_split, in_key, out_key in dialog_sources:
        try:
            meddialog = load_dataset(ds_name, split=ds_split)
            dialog_docs = [
                "[Doctor-Patient Conversation]\nPatient: " + str(i.get(in_key, "")) + "\nDoctor: " + str(i.get(out_key, ""))
                for i in meddialog if i.get(in_key) and i.get(out_key)
            ]
            if dialog_docs:
                break
        except Exception as e:
            print("Dialog dataset " + ds_name + " failed:", e)
            continue

    if not pubmed_docs and not dialog_docs:
        print("All external datasets failed. Using minimal fallback corpus.")
        pubmed_docs = [
            "[PubMed Research]\nQuestion: What causes headaches?\nAnswer: Headaches can be caused by tension, dehydration, stress, migraines, or underlying conditions like hypertension. Sudden severe headaches warrant emergency evaluation to rule out subarachnoid haemorrhage or meningitis.",
            "[PubMed Research]\nQuestion: What are symptoms of a heart attack?\nAnswer: Chest pain or pressure, radiating to arm/jaw, shortness of breath, sweating, and nausea. These require emergency care.",
            "[PubMed Research]\nQuestion: How is hypertension managed?\nAnswer: Through lifestyle changes (diet, exercise, weight loss) and medications like ACE inhibitors, beta-blockers, or calcium channel blockers. Regular monitoring is essential.",
        ]

    docs = pubmed_docs + dialog_docs
    if not docs:
        raise RuntimeError("No medical documents could be loaded. Check network connectivity.")

    embeddings = embedder.encode(docs)
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings.astype("float32"))
    return embedder, idx, docs

with st.spinner("Loading MediChat knowledge base..."):
    try:
        embedder, index, documents = load_rag_system()
    except Exception as e:
        st.error("MediChat knowledge base failed to load. Please refresh the page in a moment. Error: " + str(e))
        st.stop()

def encode_image(f):
    img = Image.open(f)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_pdf_text(uploaded_file):
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return ""
    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        text_parts = []
        for page in reader.pages[:20]:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
        full_text = "\n".join(text_parts).strip()
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "\n\n[Document truncated for length]"
        return full_text
    except Exception as e:
        print("PDF extraction failed:", e)
        return ""

def medichat_pdf_analysis(question, pdf_text, all_messages, lang_instruction=""):
    memory = extract_patient_memory(all_messages)
    memory_context = build_memory_context(memory)

    system = (
        "You are MediChat, a warm and clinically competent AI health companion. "
        "A patient has uploaded a medical PDF report (likely a blood test, lab result, or clinical letter). "
        "Your job is to read the report, identify any abnormal values, explain what they mean in plain language, "
        "and highlight what the patient should pay attention to.\n\n"
        "RULES:\n"
        "- Never use em-dashes or en-dashes. Use commas, semicolons, or periods.\n"
        "- Be warm, conversational, and clear. Avoid jargon. Explain medical terms in plain words.\n"
        "- Highlight ABNORMAL values clearly first, then briefly summarise normal results.\n"
        "- If everything is normal, say so positively.\n"
        "- Suggest specific, actionable next steps based on what's abnormal.\n"
        "- Mention patient by name if known, sparingly.\n"
        "- One brief disclaimer at most. The app shows a permanent disclaimer.\n\n"
        "FORMATTING RULES (CRITICAL for readability):\n"
        "- DO NOT use markdown headings (no #, ##, ###). They render badly in chat bubbles.\n"
        "- For section labels, write a short bold line like **What stands out:** on its own line, then continue on a new line.\n"
        "- Always put a blank line between sections.\n"
        "- Use **bold** sparingly for key values, drug names, or label headers.\n"
        "- Use bullet points (- item) for lists, each on its own line.\n"
        "- Keep paragraphs short, 2-3 sentences max.\n\n"
    )
    if lang_instruction:
        system += lang_instruction + "\n\n"
    if memory_context:
        system += "What this patient has told you in conversation:\n" + memory_context + "\n\n"

    system += "PATIENT'S UPLOADED REPORT:\n" + pdf_text + "\n\n"
    system += "PATIENT'S QUESTION ABOUT THE REPORT:\n" + (question or "Please review this report and tell me what stands out.")

    if CLAUDE_ACTIVE:
        try:
            resp = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": question or "Please review this report and tell me what stands out."}],
                temperature=0.4,
            )
            return resp.content[0].text, "claude"
        except Exception as e:
            print("Claude PDF analysis failed, falling back to Groq:", e)

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question or "Please review this report and tell me what stands out."}]
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        temperature=0.4,
        max_tokens=1500,
    )
    return r.choices[0].message.content, "groq"

def extract_patient_memory(messages):
    memory = {"symptoms": [], "conditions": [], "medications": []}
    symptom_phrases = ["i have","i feel","i am feeling","i've been feeling","i'm experiencing","i suffer from","my back hurts","my chest","my stomach","i feel pain","feeling dizzy","feeling nauseous","i have a fever","i have a cough","i have a headache","shortness of breath","i've been tired","i feel tired","i have pain","i feel sick"]
    condition_phrases = ["i have diabetes","i have hypertension","i have asthma","i have cancer","i am diabetic","i am hypertensive","i was diagnosed with","i have high blood pressure","i have low blood pressure","i have depression","i have anxiety","i have heart","i have kidney"]
    medication_phrases = ["i am taking","i take","i was prescribed","i'm on","taking medication","prescribed me","i have an inhaler","i take tablets","i take pills"]
    for msg in messages:
        if msg.get("role") == "user" and msg.get("type") == "text":
            content = msg["content"].lower()
            for phrase in symptom_phrases:
                if phrase in content:
                    pos = content.find(phrase)
                    snippet = content[pos:pos+60].strip()
                    if snippet and snippet not in memory["symptoms"]:
                        memory["symptoms"].append(snippet)
                    break
            for phrase in condition_phrases:
                if phrase in content:
                    pos = content.find(phrase)
                    snippet = content[pos:pos+60].strip()
                    if snippet and snippet not in memory["conditions"]:
                        memory["conditions"].append(snippet)
                    break
            for phrase in medication_phrases:
                if phrase in content:
                    pos = content.find(phrase)
                    snippet = content[pos:pos+60].strip()
                    if snippet and snippet not in memory["medications"]:
                        memory["medications"].append(snippet)
                    break
    return memory

def build_memory_context(memory):
    parts = []
    if memory["symptoms"]:
        parts.append("Reported symptoms: " + ", ".join(memory["symptoms"]))
    if memory["conditions"]:
        parts.append("Mentioned conditions: " + ", ".join(memory["conditions"]))
    if memory["medications"]:
        parts.append("Referenced medications: " + ", ".join(memory["medications"]))
    return "\n".join(parts)

EMERGENCY_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die", "self harm",
    "can't breathe", "cant breathe", "unable to breathe", "not breathing", "struggling to breathe",
    "chest pain", "crushing chest", "heart attack", "heart racing dangerously",
    "stroke", "having a stroke", "face drooping", "slurred speech",
    "unconscious", "passed out", "fainted", "blacking out", "lose consciousness",
    "severe bleeding", "heavy bleeding", "bleeding a lot", "cannot stop bleeding",
    "overdose", "poisoned", "took too many pills",
    "choking",
    "severe allergic reaction", "anaphylaxis", "throat closing",
    "seizure", "having a seizure", "convulsing",
    "cannot move", "can't move", "paralysed", "paralyzed",
]

EMERGENCY_SYMPTOM_CLUSTERS = [
    {
        "name": "Possible cardiac event",
        "required": [["chest", "chest pressure", "chest tight"]],
        "any_of": [["breath", "breathless", "out of breath"], ["arm pain", "jaw pain", "left arm"], ["sweat", "sweating"], ["nausea", "vomit"]],
        "min_any": 2,
    },
    {
        "name": "Possible stroke",
        "required": [],
        "any_of": [["face droop", "drooping"], ["slurred", "speech"], ["weakness one side", "numb one side"], ["sudden confusion"], ["sudden severe headache"]],
        "min_any": 2,
    },
    {
        "name": "Possible severe allergic reaction",
        "required": [],
        "any_of": [["swelling"], ["throat", "tongue"], ["hives", "rash all over"], ["difficulty breathing", "trouble breathing"]],
        "min_any": 3,
    },
    {
        "name": "Possible syncope/pre-syncope episode",
        "required": [],
        "any_of": [["faint", "faintish", "feel faint"], ["dizzy", "lightheaded"], ["blank", "going blank"], ["nausea"], ["heart racing", "heartbeat raises", "palpitations"], ["breath", "breathless"]],
        "min_any": 3,
    },
    {
        "name": "Possible severe asthma exacerbation",
        "required": [["asthma", "asthmatic"]],
        "any_of": [["can't breathe", "cant breathe", "struggling to breathe"], ["inhaler not working", "inhaler isn't helping"], ["blue lips", "blue fingers"], ["cannot speak", "can't talk"]],
        "min_any": 1,
    },
    {
        "name": "Possible hyperglycemic crisis",
        "required": [["diabetes", "diabetic"]],
        "any_of": [["excessive thirst"], ["frequent urination"], ["confusion"], ["fruity breath"], ["vomiting"], ["extreme fatigue"]],
        "min_any": 3,
    },
]

def detect_emergency(text, conversation_text=""):
    if not text and not conversation_text:
        return False, None
    combined = (text + " " + conversation_text).lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in combined:
            return True, "Emergency keyword detected"
    for cluster in EMERGENCY_SYMPTOM_CLUSTERS:
        required_met = True
        for req_group in cluster["required"]:
            if not any(term in combined for term in req_group):
                required_met = False
                break
        if not required_met:
            continue
        matches = 0
        for any_group in cluster["any_of"]:
            if any(term in combined for term in any_group):
                matches += 1
        if matches >= cluster["min_any"]:
            return True, cluster["name"]
    return False, None

DRUG_INTERACTIONS = {
    "antihistamine": {
        "drugs": ["dramamine", "dimenhydrinate", "bonine", "meclizine", "benadryl", "diphenhydramine", "chlorpheniramine", "promethazine"],
        "conditions": ["asthma", "copd", "glaucoma", "bph", "enlarged prostate"],
        "warning": "Antihistamines can dry respiratory secretions (worsening asthma/COPD), raise intraocular pressure (glaucoma), or cause urinary retention (BPH)."
    },
    "nsaid": {
        "drugs": ["ibuprofen", "advil", "nurofen", "naproxen", "aleve", "aspirin", "diclofenac", "voltaren"],
        "conditions": ["hypertension", "high blood pressure", "kidney", "renal", "ulcer", "gastritis", "asthma", "heart failure", "anticoagulant", "warfarin", "blood thinner"],
        "warning": "NSAIDs can raise blood pressure, worsen kidney function, cause GI bleeding (especially with ulcers or blood thinners), and trigger bronchospasm in some asthmatics."
    },
    "decongestant": {
        "drugs": ["pseudoephedrine", "sudafed", "phenylephrine", "oxymetazoline"],
        "conditions": ["hypertension", "high blood pressure", "heart disease", "arrhythmia", "thyroid", "hyperthyroid", "diabetes", "glaucoma", "bph"],
        "warning": "Decongestants raise blood pressure and heart rate, destabilise cardiac rhythm, worsen hyperthyroidism, and can affect glaucoma or urinary retention."
    },
    "paracetamol_high_dose": {
        "drugs": ["paracetamol", "acetaminophen", "panadol", "tylenol"],
        "conditions": ["liver", "hepatitis", "cirrhosis", "heavy alcohol", "alcoholic"],
        "warning": "High-dose or chronic paracetamol use can cause severe liver damage, especially with pre-existing liver disease or heavy alcohol use."
    },
    "bismuth": {
        "drugs": ["pepto-bismol", "bismuth subsalicylate"],
        "conditions": ["aspirin allergy", "kidney", "renal", "bleeding disorder", "anticoagulant", "warfarin"],
        "warning": "Bismuth subsalicylate contains salicylate (aspirin-like), unsafe with aspirin allergy, kidney disease, or blood thinners."
    },
    "ppi_h2": {
        "drugs": ["omeprazole", "esomeprazole", "pantoprazole", "ranitidine", "famotidine", "zantac"],
        "conditions": ["osteoporosis", "kidney", "c. diff", "magnesium deficiency"],
        "warning": "Long-term PPI use can reduce calcium/magnesium absorption (bone risk), affect kidney function, and increase C. diff infection risk."
    },
}

def check_drug_interactions(response_text, memory):
    if not response_text:
        return []
    text_lower = response_text.lower()
    conditions_text = " ".join(memory.get("conditions", [])).lower() + " " + " ".join(memory.get("medications", [])).lower()
    if not conditions_text.strip():
        return []
    alerts = []
    for class_name, info in DRUG_INTERACTIONS.items():
        drug_hit = next((d for d in info["drugs"] if d in text_lower), None)
        if not drug_hit:
            continue
        condition_hits = [c for c in info["conditions"] if c in conditions_text]
        if condition_hits:
            alerts.append({
                "drug": drug_hit.title(),
                "conditions": condition_hits,
                "warning": info["warning"]
            })
    return alerts

def calculate_confidence(distances):
    if not distances or len(distances) == 0:
        return "low", 0
    avg_dist = sum(distances) / len(distances)
    if avg_dist < 0.8:
        return "high", round((1 - avg_dist / 2) * 100)
    elif avg_dist < 1.3:
        return "medium", round((1 - avg_dist / 2) * 100)
    else:
        return "low", max(20, round((1 - avg_dist / 2.5) * 100))

def get_sources_used(idxs):
    pubmed_count = sum(1 for i in idxs if i < 500)
    dialog_count = sum(1 for i in idxs if i >= 500)
    sources = []
    if pubmed_count > 0:
        sources.append("PubMed Research (" + str(pubmed_count) + ")")
    if dialog_count > 0:
        sources.append("Doctor-Patient Data (" + str(dialog_count) + ")")
    return sources

def medichat_rag(question, all_messages, lang_instruction="", patient_name=""):
    emb = embedder.encode([question]).astype("float32")
    distances, idxs = index.search(emb, k=3)
    context = "\n\n---\n\n".join([documents[i] for i in idxs[0]])
    sources = get_sources_used(idxs[0])
    confidence_level, confidence_pct = calculate_confidence(distances[0].tolist())
    memory = extract_patient_memory(all_messages)
    memory_context = build_memory_context(memory)
    history = []
    for m in all_messages[-10:]:
        if m.get("type") == "text":
            history.append({"role": m["role"], "content": m["content"]})

    system = (
        "You are MediChat, a clinically competent AI health assistant. "
        "Patients often come to you after doctors have dismissed their concerns. "
        "Your job is to reason like a skilled GP: integrate the full symptom picture, "
        "identify the most likely diagnosis, and give genuinely useful guidance.\n\n"
    )
    if patient_name:
        system += "The patient's name is " + patient_name + ". Use their name sparingly, maximum once per response.\n\n"
    if lang_instruction:
        system += lang_instruction + "\n\n"
    if memory_context:
        system += "WHAT THIS PATIENT HAS TOLD YOU ALREADY (ANCHOR ON THIS):\n" + memory_context + "\n\n"
    system += "MEDICAL KNOWLEDGE CONTEXT (from PubMed and real doctor-patient conversations):\n" + context

    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}]
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        temperature=0.4,
        max_tokens=1024
    )
    return r.choices[0].message.content, memory, sources, confidence_level, confidence_pct

def sanitize_rag_context(raw_context):
    if not raw_context:
        return ""
    leaked_meds = [
        "subutex", "neurontin", "gabapentin", "remeron", "mirtazapine",
        "zoloft", "sertraline", "klonopin", "clonazepam", "synthroid",
        "levothyroxine", "xanax", "prozac", "lexapro", "wellbutrin",
        "lisinopril", "metformin", "atorvastatin", "amlodipine",
    ]
    text = raw_context
    patterns_to_strip = [
        r"(?i)i['']?m on [^.]*?\.",
        r"(?i)i take [^.]*?\.",
        r"(?i)i am taking [^.]*?\.",
        r"(?i)i have been on [^.]*?\.",
        r"(?i)i was prescribed [^.]*?\.",
        r"(?i)my medications? (are|include)[^.]*?\.",
        r"(?i)currently on [^.]*?\.",
    ]
    for pat in patterns_to_strip:
        text = re.sub(pat, "", text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = []
    for s in sentences:
        s_lower = s.lower()
        if any(m in s_lower for m in leaked_meds):
            continue
        clean.append(s)
    result = " ".join(clean).strip()
    return result if len(result) > 100 else raw_context

def markdown_to_html(text):
    if not text:
        return ""

    raw = text
    raw = re.sub(r"^#{1,6}\s+(.+?)$", lambda m: "\n\n**" + m.group(1).strip().rstrip(":") + ":**\n", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*#{2,6}\s+", " ", raw)
    raw = re.sub(r"(?<=[.!?:\)])\s+\*\s+(?=[A-Z])", "\n- ", raw)
    raw = re.sub(r"\*\*\s*\*\s+(?=[A-Z])", "**\n- ", raw)
    raw = re.sub(r"(?<=[a-zA-Z\.])\s+\*\s+(?=[A-Z])", "\n- ", raw)
    raw = re.sub(r"(?<=[.!?])\s+(\*\*[^\*\n]{2,40}:\*\*)", r"\n\n\1\n\n", raw)
    raw = re.sub(r"(?<=[a-zA-Z\.\)])\s+(?=-\s+[A-Z])", r"\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    try:
        import markdown as _md
        html_text = _md.markdown(raw)
    except ImportError:
        import html as _html
        safe = _html.escape(raw)
        html_text = "<p>" + safe.replace("\n\n", "</p><p>").replace("\n", "<br>") + "</p>"
        html_text = re.sub(r"\*\*([^\*\n]+)\*\*", r"<strong>\1</strong>", html_text)

    html_text = html_text.replace("<p>", "<p class='md-p'>")
    html_text = html_text.replace("<ul>", "<ul class='md-ul'>")
    html_text = html_text.replace("<ol>", "<ol class='md-ol'>")
    html_text = html_text.replace("<hr>", "<hr class='md-hr'/>")
    html_text = html_text.replace("<hr />", "<hr class='md-hr'/>")
    for level in range(1, 7):
        html_text = re.sub(
            r"<h" + str(level) + r"[^>]*>(.*?)</h" + str(level) + r">",
            r"<p class='md-p'><strong>\1</strong></p>",
            html_text,
            flags=re.DOTALL
        )
    return html_text

def strip_excessive_disclaimers(text):
    if not text:
        return text

    patterns = [
        r"\*?\*?Disclaimer:[^\n]*\*?\*?\s*",
        r"\*?\*?Please note:[^\n]*not a doctor[^\n]*\*?\*?\s*",
        r"Please note that I'?m not a doctor[^.]*\.\s*",
        r"I'?m not a doctor[,.]?[^.]*\.\s*",
        r"(I want to|I'd like to) emphasi[sz]e that I'?m not a doctor[^.]*\.\s*",
        r"(My|This) response is not a substitute for professional medical (advice|diagnosis)[^.]*\.\s*",
        r"This (conversation|response) is (not a substitute|for general information)[^.]*\.\s*",
        r"Always consult (with )?a (qualified )?(healthcare|medical) professional[^.]*\.\s*",
        r"Please consult (with )?a (qualified )?(healthcare|medical) professional[^.]*\.\s*",
        r"It'?s (important|essential|crucial) to (consult|speak with) a (doctor|healthcare professional|medical professional)[^.]*\.\s*",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+[\u2014\u2013]\s+", ", ", cleaned)
    cleaned = cleaned.replace("\u2014", ", ")
    cleaned = cleaned.replace("\u2013", ", ")
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned

def medichat_rag_stream(question, all_messages, lang_instruction="", patient_name="", pdf_context="", image_context=""):
    emb = embedder.encode([question]).astype("float32")
    distances, idxs = index.search(emb, k=3)
    raw_context = "\n\n---\n\n".join([documents[i] for i in idxs[0]])
    clean_context = sanitize_rag_context(raw_context)
    sources = get_sources_used(idxs[0])
    confidence_level, confidence_pct = calculate_confidence(distances[0].tolist())
    memory = extract_patient_memory(all_messages)
    memory_context = build_memory_context(memory)

    history = []
    for m in all_messages[-12:]:
        if m.get("type") == "text":
            history.append({"role": m["role"], "content": m["content"]})

    last_user_message = question.lower() if question else ""
    dissatisfaction_signals = [
        "not helping", "isn't helping", "not useful", "doesn't help",
        "you are not helping", "keep repeating", "already said", "same thing",
        "useless", "unhelpful", "rude", "cold",
    ]
    escalation_needed = any(sig in last_user_message for sig in dissatisfaction_signals)
    tone_complaint = any(w in last_user_message for w in ["rude", "cold", "robotic", "unfriendly"])

    system = (
        "You are MediChat — a warm, thoughtful AI health companion. "
        "You care about the person in front of you. You speak like a caring GP who happens to also be a good friend: "
        "kind, genuinely interested, never robotic, never preachy.\n\n"

        "HARD RULES (NEVER BREAK THESE)\n\n"

        "RULE 1 — NEVER FABRICATE PATIENT HISTORY:\n"
        "The <patient_history> block below shows EXACTLY what this patient has told you. "
        "The <reference_knowledge> block is generic medical information — it does NOT describe this patient. "
        "You must NEVER say 'you mentioned', 'you said', 'you told me', or 'you're taking' unless "
        "the thing you're referencing appears LITERALLY in <patient_history> or in the conversation turns below.\n\n"

        "RULE 2 — RED FLAG SCREENING FIRST:\n"
        "For these presentations, screen for danger signs BEFORE general advice:\n"
        "Sudden/severe headache, chest pain, sudden SOB, severe abdominal pain, neuro symptoms.\n"
        "If red flags present: advise emergency care clearly and without hedging.\n\n"

        "RULE 3 — NO REPETITION:\n"
        "Look at your own previous messages in this conversation. Never repeat the same advice twice. "
        "Each response must add something new.\n\n"

        "RULE 4 — ONE DISCLAIMER MAX:\n"
        "The app already shows a disclaimer. Only add 'consult a doctor' phrasing when it's genuinely the most important thing to say. "
        "Do NOT end every response with a disclaimer.\n\n"

        "RULE 5 — NO EM-DASHES OR EN-DASHES:\n"
        "Never use em-dashes (\u2014) or en-dashes (\u2013) in your responses. Use commas, semicolons, colons, periods.\n\n"

        "RULE 6 — NO MARKDOWN HEADINGS:\n"
        "Never use # ## ### markdown headings. If you need a section label, write it as a short bold line: **Section label:** then continue on a new line.\n\n"

        "RULE 7 — NO REPEATED GREETINGS OR RESTATEMENTS:\n"
        "ONLY greet the patient in your VERY FIRST message. On every subsequent turn, jump straight into your answer.\n"
        "NEVER start with 'Hi there', 'Hi [name]', 'Hello', or any greeting after the first turn.\n"
        "NEVER start responses with 'I can see you've uploaded...', 'Looking at your blood work...', 'Based on your report...', 'Thanks for sharing...'\n\n"

        "RULE 8 — MATCH RESPONSE LENGTH TO QUESTION:\n"
        "Short casual questions get short conversational answers (1-3 sentences). "
        "Long detailed questions get structured answers with bold labels.\n\n"

        "RULE 9 — REMEMBER THE CONVERSATION:\n"
        "If the patient already shared something, DO NOT ask them to repeat it.\n\n"
    )

    if tone_complaint:
        system += (
            "TONE FEEDBACK DETECTED:\n"
            "The patient has told you your tone felt off. "
            "Apologize briefly and genuinely in ONE short sentence, then show warmth.\n\n"
        )

    if escalation_needed:
        system += (
            "ESCALATION TRIGGER:\n"
            "The patient said your previous responses weren't helpful. This response must be noticeably different: "
            "more specific, more committed, with a concrete intervention.\n\n"
        )

    if patient_name:
        system += "Patient's first name: " + patient_name + ". Use naturally and sparingly (max once per response).\n\n"
    if lang_instruction:
        system += lang_instruction + "\n\n"

    system += "<patient_history>\n"
    if memory_context:
        system += "What this patient has explicitly told you in this conversation:\n" + memory_context + "\n"
    else:
        system += "This patient has NOT stated any conditions, medications, or chronic illnesses yet. Do not assume any exist.\n"
    system += "</patient_history>\n\n"

    if pdf_context:
        system += (
            "<recent_uploaded_report>\n"
            "IMPORTANT: The patient JUST uploaded this medical report and you already analyzed it. "
            "ANY question they ask now is likely a follow-up to this report. "
            "REPORT CONTENT:\n"
            + pdf_context[:6000]
            + "\n</recent_uploaded_report>\n\n"
        )

    if image_context:
        system += (
            "<recent_uploaded_image>\n"
            "IMPORTANT: The patient JUST uploaded a medical image and you already analyzed it visually. "
            "ANY question they ask now is likely a follow-up about that image. "
            "YOUR EARLIER VISUAL ANALYSIS:\n"
            + image_context[:3000]
            + "\n</recent_uploaded_image>\n\n"
        )

    system += (
        "<reference_knowledge>\n"
        "The following is generic medical information retrieved to help you reason about this query. "
        "It is NOT about the current patient.\n\n"
        + clean_context + "\n"
        "</reference_knowledge>\n"
    )

    full_response = ""
    stream_error = None

    if CLAUDE_ACTIVE:
        try:
            anthropic_messages = history + [{"role": "user", "content": question}]
            normalized = []
            for m in anthropic_messages:
                role = m["role"]
                if role not in ("user", "assistant"):
                    continue
                content = m.get("content", "")
                if normalized and normalized[-1]["role"] == role:
                    normalized[-1]["content"] += "\n\n" + content
                else:
                    normalized.append({"role": role, "content": content})
            if not normalized or normalized[0]["role"] != "user":
                normalized = [{"role": "user", "content": question}]

            with anthropic_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=system,
                messages=normalized,
                temperature=0.55,
            ) as claude_stream:
                for text in claude_stream.text_stream:
                    if text:
                        full_response += text
                        yield ("chunk", text, full_response)
            yield ("done", full_response, {"memory": memory, "sources": sources, "confidence": confidence_level, "confidence_pct": confidence_pct, "engine": "claude"})
            return
        except Exception as e:
            stream_error = e
            print("Claude stream failed, falling back to Groq:", e)
            full_response = ""

    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}]
    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        temperature=0.55,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            full_response += delta
            yield ("chunk", delta, full_response)
    yield ("done", full_response, {"memory": memory, "sources": sources, "confidence": confidence_level, "confidence_pct": confidence_pct, "engine": "groq"})

def medichat_vision(question, b64, all_messages, lang_instruction=""):
    memory = extract_patient_memory(all_messages)
    memory_context = build_memory_context(memory)
    prompt = question.strip() if question.strip() else "Please analyse this medical image."
    memory_note = ("\n\nPatient context: " + memory_context) if memory_context else ""
    lang_note = ("\n\n" + lang_instruction) if lang_instruction else ""

    system_text = (
        "You are MediChat, a warm clinical AI companion. The patient has shared a medical image (skin condition, "
        "X-ray, rash, mole, wound, scan, etc.). Analyse it carefully and respond in plain language.\n\n"
        "STRUCTURE YOUR RESPONSE WITH THESE BOLD SECTIONS:\n"
        "**What I see:**\n"
        "(2-3 sentences describing visual findings in plain language)\n\n"
        "**What this could suggest:**\n"
        "(Possible conditions, hedged appropriately.)\n\n"
        "**What you should do:**\n"
        "(Clear next steps.)\n\n"
        "FORMATTING RULES:\n"
        "- Never use em-dashes or en-dashes. Use commas, semicolons, colons.\n"
        "- Never use # ## ### markdown headings.\n"
        "- Use bullet points (- item) on their own lines for lists.\n"
        "- One brief disclaimer at most."
        + memory_note + lang_note
    )

    if CLAUDE_ACTIVE:
        try:
            resp = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1500,
                system=system_text,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                temperature=0.4,
            )
            return resp.content[0].text, "claude"
        except Exception as e:
            print("Claude vision failed, falling back to Groq:", e)

    r = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": system_text + "\n\nQuestion: " + prompt},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}
        ]}],
        temperature=0.5, max_tokens=1024
    )
    return r.choices[0].message.content, "groq-vision"

def clean_text(text):
    replacements = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2022": "-"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")

def generate_doctor_visit_summary(messages, patient_name=""):
    if not messages:
        return None, None

    transcript_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            transcript_parts.append("Patient said: " + content)
        elif role == "assistant":
            transcript_parts.append("MediChat replied: " + content)
    transcript = "\n\n".join(transcript_parts)

    name_line = ("Patient name: " + patient_name + "\n") if patient_name and patient_name != "Guest" else ""

    system_prompt = (
        "You are a medical scribe assistant. The patient had a conversation with an AI health companion. "
        "Your job is to produce a CONCISE, STRUCTURED summary the patient can hand to their actual GP at their next appointment. "
        "Use ONLY information from the conversation. Do NOT invent symptoms, dates, or details.\n\n"
        "OUTPUT FORMAT (use these exact section labels with bold markdown):\n\n"
        "**Patient overview:**\n"
        "(One sentence describing who this is and what brought them to the chat)\n\n"
        "**Symptoms reported:**\n"
        "- (each symptom on its own bullet)\n\n"
        "**Duration and pattern:**\n"
        "(When symptoms started, how often, what triggers them. If unknown, say 'not discussed'.)\n\n"
        "**Relevant medical history mentioned:**\n"
        "- (each condition or medication on its own bullet, or 'None mentioned' if not discussed)\n\n"
        "**Concerns raised by patient:**\n"
        "(What the patient is worried about, in their own framing)\n\n"
        "**MediChat's preliminary assessment:**\n"
        "(Brief summary of what the AI suggested, hedged appropriately)\n\n"
        "**Suggested questions for the GP:**\n"
        "- (3-5 specific questions the patient should ask their doctor)\n\n"
        "RULES:\n"
        "- Never use em-dashes or en-dashes. Use commas, semicolons, colons.\n"
        "- Never use # ## ### markdown headings.\n"
        "- Be factual. No fluff. No greetings. No sign-offs.\n"
    )

    user_prompt = (
        name_line +
        "Generate a doctor visit prep summary from this conversation:\n\n" +
        transcript[:8000]
    )

    summary_text = ""
    if CLAUDE_ACTIVE:
        try:
            resp = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.3,
            )
            summary_text = resp.content[0].text
        except Exception as e:
            print("Claude summary failed, falling back to Groq:", e)

    if not summary_text:
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            summary_text = r.choices[0].message.content
        except Exception as e:
            print("Groq summary failed:", e)
            return None, None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    pdf.set_fill_color(33, 118, 174)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_y(8)
    pdf.cell(0, 10, "Doctor Visit Summary", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 7, "Prepared by MediChat for: " + (patient_name if patient_name and patient_name != "Guest" else "Patient"), ln=True, align="C")
    pdf.set_y(28)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Generated on " + datetime.now().strftime("%B %d, %Y at %I:%M %p"), ln=True, align="C")

    pdf.set_y(46)

    pdf.set_fill_color(237, 246, 252)
    pdf.set_text_color(20, 66, 114)

    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, "Bring this summary to your GP appointment. It captures what you discussed with MediChat, including symptoms, history, and questions to ask. This is NOT a diagnosis. Your GP will make the clinical judgement.", fill=True, border=0)
    pdf.ln(4)

    pdf.set_text_color(40, 40, 40)
    cleaned_summary = clean_text(summary_text)

    lines = cleaned_summary.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pdf.ln(2)
            continue
        bold_match = re.match(r"^\*\*(.+?)\*\*:?$", stripped)
        if bold_match:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 58, 54)
            pdf.cell(0, 6, bold_match.group(1).rstrip(":") + ":", ln=True)
            pdf.set_text_color(40, 40, 40)
            continue
        bullet_match = re.match(r"^[\-\*]\s+(.+)$", stripped)
        if bullet_match:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(25)
            pdf.cell(5, 5, "-", ln=False)
            pdf.multi_cell(0, 5, bullet_match.group(1))
            continue
        pdf.set_font("Helvetica", "", 10)
        plain = re.sub(r"\*\*([^\*]+)\*\*", r"\1", stripped)
        pdf.multi_cell(0, 5, plain)

    pdf.ln(8)
    pdf.set_draw_color(168, 197, 189)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, "MediChat is a research prototype, not a medical device. This summary is generated by AI from a chat conversation and may contain errors or omissions. Always rely on your qualified healthcare professional for diagnosis and treatment decisions.")
    pdf.ln(2)
    pdf.cell(0, 4, "MediChat v3.0 | ICT654 Group 7 | SISTC Melbourne 2026", ln=True, align="C")

    pdf_bytes = bytes(pdf.output(dest="S"))
    return pdf_bytes, summary_text

def generate_chat_pdf(messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    pdf.set_fill_color(33, 118, 174)
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_y(8)
    pdf.cell(0, 10, "MediChat - Conversation Export", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 7, "Generated: " + datetime.now().strftime("%B %d, %Y at %I:%M %p"), ln=True, align="C")
    pdf.set_y(43)
    pdf.set_text_color(146, 64, 14)
    pdf.set_fill_color(255, 251, 235)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 5, "DISCLAIMER: This conversation is for informational purposes only. Always consult a qualified healthcare professional.", fill=True)
    pdf.ln(4)
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        msg_type = msg.get("type", "text")
        if not content:
            continue
        if role == "user":
            pdf.set_fill_color(42, 143, 197)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, "  You", ln=True, fill=True)
            pdf.set_fill_color(237, 246, 252)
            pdf.set_text_color(12, 45, 72)

            pdf.set_font("Helvetica", "", 9)
            display = "[Medical image uploaded]" + (" - " + clean_text(content) if content else "") if msg_type == "image" else clean_text(content)
            pdf.multi_cell(0, 6, "  " + display, fill=True)
        else:
            pdf.set_fill_color(30, 41, 59)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, "  MediChat", ln=True, fill=True)
            pdf.set_fill_color(248, 250, 252)
            pdf.set_text_color(51, 65, 85)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 6, "  " + clean_text(content), fill=True)
        pdf.ln(3)
    pdf.set_y(-18)
    pdf.set_text_color(148, 163, 184)
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(0, 5, "MediChat v3.0 - ICT654 Group 7 - SISTC Melbourne 2026", align="C")
    return bytes(pdf.output())

def generate_assessment_pdf(parsed, data, report_date):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    pdf.set_fill_color(33, 118, 174)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_y(7)
    pdf.cell(0, 10, "MediChat Assessment Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, "Pre-Consultation Health Summary", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 7, "Generated: " + report_date, ln=True, align="C")
    pdf.set_y(48)
    urgency = parsed.get("urgency", "See a doctor soon")
    urgency_lower = urgency.lower()
    if "emergency" in urgency_lower or "now" in urgency_lower:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_text_color(153, 27, 27)
    elif "urgent" in urgency_lower or "today" in urgency_lower:
        pdf.set_fill_color(255, 251, 235)
        pdf.set_text_color(146, 64, 14)
    else:
        pdf.set_fill_color(240, 253, 244)
        pdf.set_text_color(22, 101, 52)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 10, "URGENCY: " + clean_text(urgency), ln=True, fill=True, align="C")
    pdf.ln(5)

    def section_header(title):
        pdf.set_fill_color(33, 118, 174)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "  " + title, ln=True, fill=True)
        pdf.ln(2)


    def info_row(label, value):
        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(51, 65, 85)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 7, "  " + label + ":", fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 7, " " + clean_text(str(value)), ln=True, fill=True)
        pdf.ln(1)

    section_header("SYMPTOMS REPORTED")
    info_row("Main Symptom", data.get("main_symptom", ""))
    info_row("Duration", data.get("duration", ""))
    info_row("Severity", data.get("severity", ""))
    info_row("Pattern", data.get("pattern", ""))
    other = data.get("other_symptoms", "")
    if other and other.lower() not in ["no", "none", "n/a", "no other symptoms"]:
        info_row("Other Symptoms", other)
    info_row("Age Group", data.get("age", ""))
    info_row("Biological Sex", data.get("gender", ""))
    pdf.ln(4)
    section_header("POSSIBLE CONDITIONS")
    pdf.set_text_color(51, 65, 85)
    pdf.set_font("Helvetica", "", 9)
    for c in parsed.get("conditions", []):
        if c.strip():
            pdf.set_fill_color(240, 253, 250)
            pdf.cell(8, 7, "", fill=True)
            pdf.cell(0, 7, "- " + clean_text(c.strip()), ln=True)
            pdf.ln(1)
    pdf.ln(3)
    section_header("WHAT TO DO NEXT")
    pdf.set_text_color(51, 65, 85)
    pdf.set_font("Helvetica", "", 9)
    for i, s in enumerate(parsed.get("next_steps", []), 1):
        if s.strip():
            pdf.set_fill_color(248, 250, 252)
            pdf.multi_cell(0, 7, str(i) + ". " + clean_text(s.strip()), fill=True)
            pdf.ln(1)
    pdf.ln(3)
    section_header("MEDICHAT SUMMARY")
    summary = parsed.get("summary", "")
    if summary:
        pdf.set_fill_color(240, 253, 250)
        pdf.set_text_color(19, 78, 74)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 7, clean_text(summary), fill=True)
    pdf.ln(5)
    pdf.set_fill_color(255, 251, 235)
    pdf.set_text_color(146, 64, 14)
    pdf.set_font("Helvetica", "B", 8)
    pdf.multi_cell(0, 6, "IMPORTANT DISCLAIMER: This report was generated by MediChat AI and does NOT constitute a medical diagnosis. Please share this with your doctor for professional evaluation and treatment.", fill=True)
    pdf.set_y(-18)
    pdf.set_text_color(148, 163, 184)
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(0, 5, "MediChat v3.0 - ICT654 Group 7 - SISTC Melbourne 2026", align="C")
    return bytes(pdf.output())

ASSESSMENT_STAGES = [
    {"key": "main_symptom", "question": "What is your main symptom or health concern today?", "hint": "Please describe clearly e.g. chest pain, headache, fever, shortness of breath, dizziness...", "options": ["Chest pain", "Headache", "Fever", "Cough", "Dizziness", "Stomach pain", "Fatigue", "Other (type below)"]},
    {"key": "duration", "question": "How long have you been experiencing this?", "hint": "", "options": ["Just started today", "A few days", "About a week", "More than 2 weeks", "Over a month"]},
    {"key": "severity", "question": "How severe is it on a scale of 1 to 10? (1 = mild, 10 = unbearable)", "hint": "", "options": ["1-2 (Mild)", "3-4 (Moderate)", "5-6 (Significant)", "7-8 (Severe)", "9-10 (Unbearable)"]},
    {"key": "pattern", "question": "Is it constant or does it come and go?", "hint": "", "options": ["Constant, always there", "Comes and goes", "Getting worse over time", "Getting better", "Only happens sometimes"]},
    {"key": "other_symptoms", "question": "Are you experiencing any other symptoms alongside this?", "hint": "e.g. nausea, dizziness, fever, fatigue... or select None", "options": ["No other symptoms", "Nausea or vomiting", "Fever or chills", "Dizziness", "Fatigue or weakness", "Other (type below)"]},
    {"key": "age", "question": "How old are you?", "hint": "", "options": ["Under 18", "18-30", "31-45", "46-60", "61-75", "Over 75"]},
    {"key": "gender", "question": "What is your biological sex? (helps with medical accuracy)", "hint": "", "options": ["Male", "Female", "Prefer not to say"]},
]

def generate_assessment_report(assessment_data, lang_instruction=""):
    emb = embedder.encode([assessment_data.get("main_symptom", "")]).astype("float32")
    _, idxs = index.search(emb, k=5)
    context = "\n\n---\n\n".join([documents[i] for i in idxs[0]])
    lang_note = ("\n" + lang_instruction) if lang_instruction else ""
    prompt = (
        "You are an experienced clinical AI assistant.\n\n"
        "Patient symptom assessment data:\n"
        "- Main symptom: " + assessment_data.get("main_symptom", "Not specified") + "\n"
        "- Duration: " + assessment_data.get("duration", "Not specified") + "\n"
        "- Severity: " + assessment_data.get("severity", "Not specified") + "\n"
        "- Pattern: " + assessment_data.get("pattern", "Not specified") + "\n"
        "- Other symptoms: " + assessment_data.get("other_symptoms", "None") + "\n"
        "- Age group: " + assessment_data.get("age", "Not specified") + "\n"
        "- Biological sex: " + assessment_data.get("gender", "Not specified") + "\n\n"
        "INSTRUCTIONS:\n"
        "1. If the main symptom looks like a typo, interpret the most likely intended symptom.\n"
        "2. Consider all symptoms holistically.\n"
        "3. Provide realistic, evidence-based possible conditions.\n"
        "4. Urgency must reflect actual severity.\n"
        + lang_note + "\n\n"
        "Medical research context:\n" + context + "\n\n"
        "Respond in EXACTLY this format:\n"
        "URGENCY: [one of: Self-care at home / See a doctor soon / Seek urgent care today / Go to emergency NOW]\n"
        "CONDITIONS: [condition 1] | [condition 2] | [condition 3]\n"
        "NEXT STEPS: [step 1] | [step 2] | [step 3]\n"
        "SUMMARY: [2-3 warm, clear sentences summarising the assessment]\n"
    )
    r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
    return r.choices[0].message.content

def parse_report(report_text):
    parsed = {"urgency": "", "conditions": [], "next_steps": [], "summary": ""}
    for line in report_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("URGENCY:"):
            parsed["urgency"] = line.replace("URGENCY:", "").strip()
        elif line.startswith("CONDITIONS:"):
            parsed["conditions"] = [c.strip() for c in line.replace("CONDITIONS:", "").split("|") if c.strip()]
        elif line.startswith("NEXT STEPS:"):
            parsed["next_steps"] = [s.strip() for s in line.replace("NEXT STEPS:", "").split("|") if s.strip()]
        elif line.startswith("SUMMARY:"):
            parsed["summary"] = line.replace("SUMMARY:", "").strip()
    return parsed

if "session_started" not in st.session_state:
    st.session_state.session_started = True
    st.session_state.messages = []
    st.session_state.qcount = 0
    st.session_state.feedback = {}
    st.session_state.patient_memory = {"symptoms": [], "conditions": [], "medications": []}
    st.session_state.uploader_key = 0
    st.session_state.last_pdf_context = ""
    st.session_state.last_pdf_name = ""
    st.session_state.last_image_context = ""
    st.session_state.mode = "chat"
    st.session_state.assessment_stage = 0
    st.session_state.assessment_data = {}
    st.session_state.assessment_complete = False
    st.session_state.assessment_report = None
    st.session_state.assessment_parsed = None
    st.session_state.selected_language = "English"
    st.session_state.patient_name = ""
    st.session_state.emergency_detected = False
    st.session_state.emergency_reason = ""
    st.session_state.last_sources = []
    st.session_state.eval_log = []
    st.session_state.response_times = []
    st.session_state.admin_authenticated = False
    st.session_state.admin_attempt_failed = False
    st.session_state.chat_input_key = 0
    st.session_state.is_authenticated = False
    st.session_state.is_guest = False
    st.session_state.user_email_hash = ""
    st.session_state.user_email_display = ""
    st.session_state.auth_error = ""
    st.session_state.auth_view = "choose"

with st.sidebar:
    st.markdown("## MediChat")
    st.markdown("---")

    if st.session_state.is_authenticated:
        _name = st.session_state.patient_name or "Patient"
        _email = st.session_state.user_email_display or ""
        _initial = (_name[0] if _name and _name != "Patient" else (_email[0] if _email else "P")).upper()
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.6rem;background:linear-gradient(135deg,#edf6fc,#d6edf9);border:1px solid #b0daf2;border-radius:12px;padding:0.6rem 0.8rem;margin-bottom:0.6rem;">'
            '<div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#2176ae,#144272);color:white;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;">' + _initial + '</div>'
            '<div style="flex:1;min-width:0;">'
            '<div style="font-size:0.8rem;font-weight:600;color:#0c2d48;line-height:1.1;">' + _name + '</div>'
            '<div style="font-size:0.65rem;color:#1a5b8a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">Profile saved</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("Sign out", use_container_width=True, key="profile_logout"):
            for k in ["is_authenticated", "is_guest", "user_email_hash", "user_email_display", "patient_name", "patient_memory", "messages", "qcount", "feedback", "last_sources", "last_pdf_context", "last_image_context"]:
                if k in st.session_state:
                    if k in ("is_authenticated", "is_guest"):
                        st.session_state[k] = False
                    elif k == "patient_memory":
                        st.session_state[k] = {"symptoms": [], "conditions": [], "medications": []}
                    elif k == "messages":
                        st.session_state[k] = []
                    elif k == "qcount":
                        st.session_state[k] = 0
                    elif k == "feedback":
                        st.session_state[k] = {}
                    else:
                        st.session_state[k] = "" if isinstance(st.session_state[k], str) else st.session_state[k]
            st.rerun()
        st.markdown("---")
    elif st.session_state.is_guest:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.5rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:0.55rem 0.75rem;margin-bottom:0.6rem;font-size:0.78rem;color:#475569;">'
            '<span>👤</span><span>Guest session — not saved</span>'
            '</div>',
            unsafe_allow_html=True
        )
        if FIREBASE_ACTIVE and st.button("Sign in / create profile", use_container_width=True, key="guest_to_signin"):
            st.session_state.is_guest = False
            st.rerun()
        st.markdown("---")

    st.markdown('<div class="sb-title">Language / மொழி / භාෂාව / भाषा</div>', unsafe_allow_html=True)
    lang_options = list(LANGUAGES.keys())
    lang_display = [LANGUAGES[l]["flag"] + " " + l for l in lang_options]
    selected_idx = lang_options.index(st.session_state.selected_language)
    chosen = st.selectbox("", lang_display, index=selected_idx, label_visibility="collapsed")
    new_lang = lang_options[lang_display.index(chosen)]
    if new_lang != st.session_state.selected_language:
        st.session_state.selected_language = new_lang
        st.rerun()

    L = LANGUAGES[st.session_state.selected_language]

    st.markdown("---")
    st.markdown('<div class="sb-title">Session Stats</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sb-stat-card"><div class="sb-stat-num">' + str(st.session_state.qcount) + '</div><div class="sb-stat-label">Questions</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="sb-stat-card"><div class="sb-stat-num">1000</div><div class="sb-stat-label">Medical Docs</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    mem = st.session_state.patient_memory
    if any([mem.get("symptoms"), mem.get("conditions"), mem.get("medications")]):
        st.markdown('<div class="sb-title">Patient Memory</div>', unsafe_allow_html=True)
        if mem.get("symptoms"):
            st.markdown('<div class="sb-memory-item">Symptoms: ' + ", ".join(mem["symptoms"][:2]) + '</div>', unsafe_allow_html=True)
        if mem.get("conditions"):
            st.markdown('<div class="sb-memory-item">Conditions: ' + ", ".join(mem["conditions"][:2]) + '</div>', unsafe_allow_html=True)
        if mem.get("medications"):
            st.markdown('<div class="sb-memory-item">Medications: ' + ", ".join(mem["medications"][:2]) + '</div>', unsafe_allow_html=True)
        st.markdown("---")
    st.markdown('<div class="sb-title">Active Features</div>', unsafe_allow_html=True)
    features = [
        ("#dc2626", "Emergency Detection"),
        ("#0d9488", "RAG Pipeline"),
        ("#7c3aed", "Vision AI"),
        ("#0369a1", "Chat Memory"),
        ("#059669", "Symptom Check"),
        ("#d97706", "PDF Export"),
        ("#0ea5e9", "Source Transparency"),
        ("#8b5cf6", "Multilingual (5)"),
    ]
    for color, name in features:
        st.markdown('<div class="sb-feature"><div class="sb-feature-dot" style="background:' + color + ';"></div><div class="sb-feature-name">' + name + '</div><div class="sb-feature-status">Live</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sb-title">Try Asking</div>', unsafe_allow_html=True)
    for tip in ["What causes high blood pressure?", "I have chest pain and I am diabetic", "How does stress affect the heart?", "What foods reduce inflammation?", "I have been dizzy since yesterday"]:
        st.markdown('<div class="sb-tip">- ' + tip + '</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sb-footer">MediChat v6.0<br>ICT654 - Group 7 - SISTC 2026</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="header-card">'
    '<div class="header-brand">'
    '<div class="header-logo">✦</div>'
    '<div>'
    '<div class="header-title">MediChat</div>'
    '<div class="header-subtitle">A thoughtful health companion, grounded in real medical research</div>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="trust-strip">'
    '<span class="trust-pill"><span class="trust-pill-icon">🔒</span>Private &amp; Confidential</span>'
    '<span class="trust-pill"><span class="trust-pill-icon">📚</span>1,000 medical sources</span>'
    '<span class="trust-pill"><span class="trust-pill-icon">✓</span>Evidence-based</span>'
    '<span class="trust-pill"><span class="trust-pill-icon">🌐</span>5 languages</span>'
    '</div>',
    unsafe_allow_html=True
)

L = LANGUAGES[st.session_state.selected_language]

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", os.environ.get("ADMIN_PASSWORD", "MediChat@Group7#2026"))
_query_params = st.query_params
_admin_requested = _query_params.get("admin", "") != ""

if _admin_requested and not st.session_state.admin_authenticated:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1f2937,#111827);color:white;padding:1.5rem 2rem;border-radius:16px;margin:2rem auto;max-width:500px;box-shadow:0 8px 30px rgba(0,0,0,0.2);">'
        '<div style="text-align:center;margin-bottom:1.2rem;">'
        '<div style="font-size:2.5rem;margin-bottom:0.5rem;">🔒</div>'
        '<div style="font-size:1.3rem;font-weight:700;margin-bottom:0.3rem;">Admin Access Required</div>'
        '<div style="font-size:0.85rem;opacity:0.8;">Research &amp; Evaluation Dashboard</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    pg_c1, pg_c2, pg_c3 = st.columns([1, 2, 1])
    with pg_c2:
        with st.form(key="admin_login_form", clear_on_submit=True):
            admin_pw_input = st.text_input("Password", type="password", placeholder="Enter admin password", label_visibility="collapsed")
            login_btn = st.form_submit_button("Unlock Analytics", use_container_width=True)
        if login_btn:
            if admin_pw_input == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.session_state.admin_attempt_failed = False
                st.rerun()
            else:
                st.session_state.admin_attempt_failed = True
                st.rerun()
        if st.session_state.admin_attempt_failed:
            st.error("Incorrect password. Access denied.")
        st.caption("Authorised team members only. This dashboard contains anonymised research data.")
    st.stop()

_is_admin = st.session_state.admin_authenticated

# ── Patient Profile Auth Gate ────────────────────────────────────────
# Only enforced when Firebase is connected and user is not admin.
# Users can also continue as Guest (no persistence).
if FIREBASE_ACTIVE and not _is_admin and not st.session_state.is_authenticated and not st.session_state.is_guest:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#144272,#0c2d48);color:white;padding:1.6rem 2rem;border-radius:18px;margin:1rem auto 1.4rem auto;max-width:560px;box-shadow:0 8px 30px rgba(12,45,72,0.18);">'
        '<div style="text-align:center;">'
        '<div style="font-size:2.2rem;margin-bottom:0.4rem;">👤</div>'
        '<div style="font-family:\'DM Serif Display\',serif;font-size:1.5rem;margin-bottom:0.3rem;">Welcome to MediChat - your virtual doctor.</div>'
        '<div style="font-size:0.88rem;opacity:0.85;line-height:1.55;">Sign in to keep your health profile across visits, or continue as a guest for a one-off MediChat.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    auth_c1, auth_c2, auth_c3 = st.columns([1, 3, 1])
    with auth_c2:
        view = st.session_state.auth_view
        if view == "choose":
            tab_signin, tab_signup, tab_guest = st.tabs(["Sign in", "Create profile", "Guest"])
            with tab_signin:
                with st.form("signin_form", clear_on_submit=False):
                    si_email = st.text_input("Email", placeholder="you@example.com", key="si_email")
                    si_pin = st.text_input("4-6 digit PIN", type="password", max_chars=6, key="si_pin")
                    si_btn = st.form_submit_button("Sign in", use_container_width=True, type="primary")
                if si_btn:
                    if not si_email or "@" not in si_email or not si_pin or not si_pin.isdigit() or len(si_pin) < 4:
                        st.error("Please enter a valid email and a 4-6 digit numeric PIN.")
                    else:
                        profile, status = authenticate_profile(si_email, si_pin)
                        if status == "ok":
                            st.session_state.is_authenticated = True
                            st.session_state.user_email_hash = profile["email_hash"]
                            st.session_state.user_email_display = si_email.strip()
                            st.session_state.patient_name = profile.get("name", "") or ""
                            st.session_state.patient_memory = profile.get("patient_memory", {"symptoms": [], "conditions": [], "medications": []})
                            st.session_state.selected_language = profile.get("language", "English")
                            restored_msgs = profile.get("messages") or []
                            if restored_msgs:
                                st.session_state.messages = restored_msgs
                                st.session_state.qcount = sum(1 for m in restored_msgs if m.get("role") == "user")
                            st.success("Welcome back" + ((", " + profile.get("name", "")) if profile.get("name") else "") + ". Loading your profile…")
                            st.rerun()
                        elif status == "not_found":
                            st.error("No profile found for that email. Switch to 'Create profile' to start one.")
                        elif status == "wrong_pin":
                            st.error("Incorrect PIN. Try again or recover your account by creating a new profile with a different email.")
            with tab_signup:
                with st.form("signup_form", clear_on_submit=False):
                    su_email = st.text_input("Email", placeholder="you@example.com", key="su_email")
                    su_name = st.text_input("First name (optional)", key="su_name", max_chars=30)
                    su_pin = st.text_input("Choose a 4-6 digit PIN", type="password", max_chars=6, key="su_pin")
                    su_pin2 = st.text_input("Confirm PIN", type="password", max_chars=6, key="su_pin2")
                    su_btn = st.form_submit_button("Create profile", use_container_width=True, type="primary")
                if su_btn:
                    if not su_email or "@" not in su_email:
                        st.error("Please enter a valid email.")
                    elif not su_pin or not su_pin.isdigit() or len(su_pin) < 4:
                        st.error("PIN must be 4-6 digits, numbers only.")
                    elif su_pin != su_pin2:
                        st.error("PINs do not match.")
                    elif get_profile(hash_email(su_email)) is not None:
                        st.error("A profile already exists for that email. Sign in instead.")
                    else:
                        profile = create_profile(su_email, su_pin, su_name)
                        if profile is None:
                            st.error("Could not create profile. Please try again in a moment.")
                        else:
                            st.session_state.is_authenticated = True
                            st.session_state.user_email_hash = profile["email_hash"]
                            st.session_state.user_email_display = su_email.strip()
                            st.session_state.patient_name = profile.get("name", "") or ""
                            st.success("Profile created. Welcome to MediChat.")
                            st.rerun()
            with tab_guest:
                st.markdown(
                    '<div style="padding:0.8rem 0;font-size:0.88rem;color:#334155;line-height:1.6;">'
                    'Guest mode lets you try MediChat without an account. Your conversation lives only for this session. '
                    'When you close the tab, everything is forgotten. Switch to a profile any time for continuity across visits.'
                    '</div>',
                    unsafe_allow_html=True
                )
                if st.button("Continue as Guest", use_container_width=True, key="go_guest_btn"):
                    st.session_state.is_guest = True
                    st.rerun()
        st.caption("Your email is stored as an irreversible hash. Your PIN is salted and hashed. Neither is reversible by the team.")
    st.stop()

if _is_admin:
    admin_c1, admin_c2 = st.columns([5, 1])
    with admin_c2:
        if st.button("🔒 Logout", key="admin_logout"):
            st.session_state.admin_authenticated = False
            st.session_state.mode = "chat"
            st.rerun()
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        if st.button(L["free_chat"] + (" (Active)" if st.session_state.mode == "chat" else ""), use_container_width=True):
            st.session_state.mode = "chat"
            st.rerun()
    with cm2:
        if st.button(L["symptom_check"] + (" (Active)" if st.session_state.mode == "assessment" else ""), use_container_width=True):
            st.session_state.mode = "assessment"
            st.rerun()
    with cm3:
        if st.button("📊 Analytics" + (" (Active)" if st.session_state.mode == "eval" else ""), use_container_width=True):
            st.session_state.mode = "eval"
            st.rerun()
else:
    if st.session_state.mode == "eval":
        st.session_state.mode = "chat"
    cm1, cm2 = st.columns(2)
    with cm1:
        if st.button(L["free_chat"] + (" (Active)" if st.session_state.mode == "chat" else ""), use_container_width=True):
            st.session_state.mode = "chat"
            st.rerun()
    with cm2:
        if st.button(L["symptom_check"] + (" (Active)" if st.session_state.mode == "assessment" else ""), use_container_width=True):
            st.session_state.mode = "assessment"
            st.rerun()

st.markdown('<div class="disclaimer">MediChat offers general health information, not personal medical advice. Please consult a qualified doctor for any health concerns that need diagnosis or treatment.</div>', unsafe_allow_html=True)

if st.session_state.emergency_detected:
    reason = st.session_state.get("emergency_reason", "Emergency indicators detected")
    st.markdown(
        '<div class="emergency-banner">'
        '<div class="emergency-title">🚨 This May Be a Medical Emergency</div>'
        '<div class="emergency-text"><strong>Detected pattern:</strong> ' + reason + '. Based on what you described, you may need immediate medical attention. Please stop and call emergency services now. Do not wait.</div>'
        '<div class="emergency-number">📞 Call 000 (Australia)</div>'
        '<div style="font-size:0.75rem;margin-top:0.5rem;opacity:0.9;">Other countries: 911 (USA) | 999 (UK) | 112 (EU) | 119 (Sri Lanka) | 102 (India)</div>'
        '</div>',
        unsafe_allow_html=True
    )
    cols = st.columns([3, 1])
    with cols[1]:
        if st.button("Dismiss alert", key="dismiss_emergency"):
            st.session_state.emergency_detected = False
            st.rerun()

if st.session_state.mode == "chat":
    mem = st.session_state.patient_memory

    if any([mem.get("symptoms"), mem.get("conditions"), mem.get("medications")]) and st.session_state.messages:
        mem_parts = []
        if mem.get("symptoms"):
            mem_parts.append("Symptoms: " + ", ".join(mem["symptoms"]))
        if mem.get("conditions"):
            mem_parts.append("Conditions: " + ", ".join(mem["conditions"]))
        if mem.get("medications"):
            mem_parts.append("Medications: " + ", ".join(mem["medications"]))
        st.markdown('<div class="memory-card"><div class="memory-title">MediChat remembers from this session:</div>' + "".join(["<div>- " + p + "</div>" for p in mem_parts]) + "</div>", unsafe_allow_html=True)

    show_hero = (not st.session_state.messages) and (not st.session_state.patient_name)

    if show_hero:
        st.markdown(
            '<div class="hero-wrap">'
            '<div class="hero-eyebrow">Live · Testing Stage</div>'
            '<div class="hero-title">Your private clinical AI companion.</div>'
            '<div class="hero-subtitle">Free. Multilingual. Evidence-grounded health guidance, with full transparency about sources and confidence.</div>'
            '<div class="hero-trust-row">'
            '<span class="trust-pill"><span class="trust-icon">🔒</span> Privacy by design</span>'
            '<span class="trust-pill"><span class="trust-icon">📚</span> 1,000 medical sources</span>'
            '<span class="trust-pill"><span class="trust-icon">🌐</span> 5 languages</span>'
            '<span class="trust-pill"><span class="trust-icon">⚡</span> Powered by Claude Haiku</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="font-size:0.88rem;color:var(--clinical-800);margin:0.3rem 0 0.5rem 0;font-weight:500;">'
            "What should I call you? <span style=\"color:var(--neutral-500);font-weight:400;font-size:0.82rem;\">(optional)</span>"
            '</div>',
            unsafe_allow_html=True
        )

        with st.form(key="name_form", clear_on_submit=True):
            name_cols = st.columns([3, 1, 1])
            with name_cols[0]:
                name_typed = st.text_input("", placeholder="Type your first name...", label_visibility="collapsed")
            with name_cols[1]:
                name_submit = st.form_submit_button("Save")
            with name_cols[2]:
                skip_name = st.form_submit_button("Skip")
        if name_submit and name_typed.strip():
            st.session_state.patient_name = name_typed.strip()[:20]
            if st.session_state.is_authenticated and st.session_state.user_email_hash:
                persist_profile_state(st.session_state.user_email_hash, name=st.session_state.patient_name)
            st.rerun()
        if skip_name:
            st.session_state.patient_name = "Guest"
            st.rerun()
    elif not st.session_state.messages and st.session_state.patient_name:
        display_name = "" if st.session_state.patient_name == "Guest" else ", " + st.session_state.patient_name
        st.markdown(
            '<div style="text-align:center;padding:1.2rem 1rem;color:var(--clinical-700);">'
            '<div style="font-family:\'DM Serif Display\',serif;font-size:1.3rem;color:var(--clinical-900);margin-bottom:0.3rem;">Hi' + display_name + ', how can I help you today?</div>'
            '<div style="font-size:0.85rem;color:var(--neutral-600);">Ask anything, upload an image or PDF report, or describe what is on your mind.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    else:
        user_initial = "U"
        if st.session_state.patient_name and st.session_state.patient_name != "Guest":
            user_initial = st.session_state.patient_name[0].upper()
        user_name_label = st.session_state.patient_name if st.session_state.patient_name and st.session_state.patient_name != "Guest" else "You"

        for msg in st.session_state.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            msg_type = msg.get("type", "text")
            if role == "user":
                if msg_type == "image":
                    st.markdown('<span class="image-tag">Medical image uploaded for analysis</span>', unsafe_allow_html=True)
                    if content:
                        st.markdown('<div class="user-wrap"><div class="user-bubble">' + content + '</div><div class="av av-user">' + user_initial + '</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="user-wrap"><div class="user-bubble">' + content + '</div><div class="av av-user">' + user_initial + '</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="bot-label">MediChat</div>', unsafe_allow_html=True)
                st.markdown('<div class="bot-wrap"><div class="av av-bot">M</div><div class="bot-bubble">' + markdown_to_html(content) + '</div></div>', unsafe_allow_html=True)
                engine_used = msg.get("engine", "")
                msg_sources = msg.get("sources", [])
                is_image_response = "Image Analysis" in msg_sources or engine_used == "groq-vision"
                is_pdf_response = "PDF Report Analysis" in msg_sources

                engine_html = ""
                if engine_used == "claude":
                    engine_html = '<span class="engine-badge engine-claude">Claude Haiku</span>'
                elif engine_used == "groq":
                    engine_html = '<span class="engine-badge engine-groq">Llama (fallback)</span>'
                elif engine_used == "groq-vision":
                    engine_html = '<span class="engine-badge engine-vision">Llama Vision (fallback)</span>'

                if is_image_response:
                    source_tags = '<span class="source-tag">📷 Image Analysis</span>'
                elif is_pdf_response:
                    source_tags = '<span class="source-tag">📄 PDF Report Analysis</span>'
                else:
                    source_tags = "".join(['<span class="source-tag">📚 ' + s + '</span>' for s in msg_sources])

                if engine_html or source_tags:
                    st.markdown('<div class="source-row">' + engine_html + source_tags + '</div>', unsafe_allow_html=True)

                conf_level = msg.get("confidence")
                conf_pct = msg.get("confidence_pct")
                if conf_level and conf_pct and not is_image_response and engine_used != "system":
                    conf_label = {"high": "High Confidence", "medium": "Medium Confidence", "low": "Low Confidence"}.get(conf_level, "")
                    conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(conf_level, "#64748b")
                    st.markdown(
                        '<div class="confidence-row">'
                        '<span class="confidence-pill conf-' + conf_level + '">' + conf_label + '</span>'
                        '<span class="confidence-bar"><span class="confidence-fill" style="width:' + str(conf_pct) + '%;background:' + conf_color + ';"></span></span>'
                        '<span style="color:#64748b;">' + str(conf_pct) + '% RAG match quality</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

    if st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:0.78rem;color:#64748b;margin-bottom:0.4rem;">' + L["helpful"] + '</div>', unsafe_allow_html=True)
        cf1, cf2, cf3, cf4, cf5 = st.columns([2, 1, 0.5, 1, 2])
        with cf2:
            if st.button(L["yes"], key="chat_helpful"):
                st.session_state.feedback["overall"] = "helpful"
                st.rerun()
        with cf4:
            if st.button(L["no"], key="chat_not_helpful"):
                st.session_state.feedback["overall"] = "not_helpful"
                st.rerun()
        overall = st.session_state.feedback.get("overall")
        if overall == "helpful":
            st.markdown('<div style="text-align:center;font-size:0.76rem;color:#0f766e;margin-top:0.3rem;">' + L["thanks_helpful"] + '</div>', unsafe_allow_html=True)
        elif overall == "not_helpful":
            st.markdown('<div style="text-align:center;font-size:0.76rem;color:#dc2626;margin-top:0.3rem;">' + L["thanks_not"] + '</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**" + L["download_chat"] + "**")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            if st.button(L["download_chat_btn"], use_container_width=True, key="dl_chat_btn"):
                pdf_bytes = generate_chat_pdf(st.session_state.messages)
                st.download_button(label="Click here to save your PDF", data=pdf_bytes, file_name="MediChat_Conversation_" + datetime.now().strftime("%Y%m%d_%H%M") + ".pdf", mime="application/pdf", use_container_width=True, key="dl_chat_dl")

        with dl_col2:
            if st.button("📋 Doctor Visit Summary", use_container_width=True, key="dl_summary_btn", help="Generates a structured one-page summary you can hand to your GP at your next appointment"):
                with st.spinner("Preparing your doctor visit summary..."):
                    summary_pdf, _summary_text = generate_doctor_visit_summary(st.session_state.messages, st.session_state.patient_name)
                if summary_pdf:
                    st.download_button(
                        label="Save Doctor Visit Summary PDF",
                        data=summary_pdf,
                        file_name="MediChat_DoctorVisitSummary_" + datetime.now().strftime("%Y%m%d_%H%M") + ".pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_summary_dl"
                    )
                else:
                    st.error("Could not generate summary. Please try again or have a longer conversation first.")

        st.markdown(
            '<div style="font-size:0.72rem;color:var(--neutral-500);margin-top:0.5rem;text-align:center;font-style:italic;">'
            'The Doctor Visit Summary is a structured one-pager you can bring to your next GP appointment.'
            '</div>',
            unsafe_allow_html=True
        )


    st.markdown('<div id="chat-input-anchor"></div>', unsafe_allow_html=True)

    # ── Text input + Send (inside form so Enter key submits) ────────────
    with st.form("chat_form", clear_on_submit=True):
        fc1, fc2 = st.columns([4, 1])
        with fc1:
            user_input = st.text_input(
                "Your message",
                placeholder=L["placeholder"],
                label_visibility="collapsed",
                key="chat_input_" + str(st.session_state.chat_input_key),
            )
        with fc2:
            submit = st.form_submit_button("➤  " + L["send_btn"], use_container_width=True, type="primary")

    # ── Button row: [📎 Attach] [Clear] ─ equal width, parallel ────────
    bc_attach, bc_clear = st.columns(2)
    with bc_attach:
        uploaded_image = st.file_uploader(
            "Attach a medical image or PDF report",
            type=["jpg", "jpeg", "png", "pdf"],
            label_visibility="collapsed",
            key="uploader_" + str(st.session_state.uploader_key),
            help="Attach a medical image (JPG, PNG) or PDF report",
        )
    with bc_clear:
        clear = st.button("🗑  " + L["clear_btn"], use_container_width=True, key="main_clear_btn")

    # ── File preview (shown below button row once attached) ─────────────
    if uploaded_image:
        is_pdf_upload = uploaded_image.name.lower().endswith(".pdf")
        if is_pdf_upload:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.7rem;padding:0.8rem 1rem;background:var(--clinical-50);border:1px solid var(--clinical-200);border-radius:12px;margin:0.5rem 0;">'
                '<div style="width:40px;height:40px;background:var(--clinical-600);color:white;border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.7rem;">PDF</div>'
                '<div style="flex:1;">'
                '<div style="font-weight:600;font-size:0.88rem;color:var(--clinical-900);">' + uploaded_image.name + '</div>'
                '<div style="font-size:0.75rem;color:var(--neutral-600);">Ready for analysis. Ask MediChat what you want to know about this report.</div>'

                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            ia, ib, ic = st.columns([1, 2, 1])
            with ib:
                st.image(uploaded_image, caption="Ready for analysis", use_column_width=True)

    st.markdown('<div class="disclaimer-mini disclaimer-mini-red">⚠ MediChat is not a substitute for professional medical advice. For diagnosis or treatment, please consult a qualified doctor.</div>', unsafe_allow_html=True)
    st.markdown('<div id="page-bottom-anchor" style="height:1px;"></div>', unsafe_allow_html=True)

    if st.session_state.messages:
        import streamlit.components.v1 as _components
        _components.html(
            """
            <script>
                (function() {
                    function scrollToBottom() {
                        try {
                            const doc = window.parent.document;
                            const anchor = doc.getElementById('page-bottom-anchor');
                            if (anchor) {
                                anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
                                return;
                            }
                            const mainBlock = doc.querySelector('section.main') ||
                                              doc.querySelector('[data-testid="stAppViewContainer"]') ||
                                              doc.querySelector('.main');
                            if (mainBlock) {
                                mainBlock.scrollTo({ top: mainBlock.scrollHeight, behavior: 'smooth' });
                            }
                            window.parent.scrollTo({ top: doc.body.scrollHeight, behavior: 'smooth' });
                        } catch (e) {}
                    }
                    setTimeout(scrollToBottom, 200);
                    setTimeout(scrollToBottom, 600);
                    setTimeout(scrollToBottom, 1200);
                })();
            </script>
            """,
            height=0,
        )

    if clear:
        st.session_state.messages = []
        st.session_state.qcount = 0
        st.session_state.feedback = {}
        st.session_state.patient_memory = {"symptoms": [], "conditions": [], "medications": []}
        st.session_state.uploader_key += 1
        st.session_state.emergency_detected = False
        st.session_state.emergency_reason = ""
        st.session_state.last_sources = []
        st.session_state.patient_name = ""
        st.session_state.last_pdf_context = ""
        st.session_state.last_pdf_name = ""
        st.session_state.last_image_context = ""
        st.session_state.chat_input_key = st.session_state.get("chat_input_key", 0) + 1
        if st.session_state.is_authenticated and st.session_state.user_email_hash:
            persist_profile_state(
                st.session_state.user_email_hash,
                patient_memory=st.session_state.patient_memory,
                messages=[],
            )
        st.rerun()

    if submit and (user_input.strip() or uploaded_image):
        st.session_state.qcount += 1
        lang_instruction = LANGUAGES[st.session_state.selected_language]["lang_instruction"]

        if user_input.strip():
            conv_text = " ".join([m.get("content", "") for m in st.session_state.messages if m.get("type") == "text"])
            is_emerg, reason = detect_emergency(user_input, conv_text)
            if is_emerg:
                st.session_state.emergency_detected = True
                st.session_state.emergency_reason = reason

        if uploaded_image:
            is_pdf = uploaded_image.name.lower().endswith(".pdf")
            if is_pdf:
                st.session_state.messages.append({"role": "user", "type": "pdf", "content": (user_input.strip() + " " if user_input.strip() else "") + "[PDF: " + uploaded_image.name + "]"})
                with st.spinner("Reading your medical report..."):
                    pdf_text = extract_pdf_text(uploaded_image)
                    if not pdf_text:
                        reply = "I had trouble reading that PDF. It might be image-based (scanned) rather than text-based. Could you try uploading it as a JPEG or PNG image instead?"
                        engine_used = "system"
                    else:
                        st.session_state.last_pdf_context = pdf_text
                        st.session_state.last_pdf_name = uploaded_image.name
                        reply, engine_used = medichat_pdf_analysis(user_input, pdf_text, st.session_state.messages, lang_instruction)
                        reply = strip_excessive_disclaimers(reply)
                st.session_state.last_sources = ["PDF Report Analysis"]
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply, "sources": st.session_state.last_sources, "confidence": "medium", "confidence_pct": 75, "engine": engine_used})
                st.session_state.uploader_key += 1
            else:
                st.session_state.messages.append({"role": "user", "type": "image", "content": user_input.strip()})
                with st.spinner("Analysing your image..."):
                    uploaded_image.seek(0)
                    reply, vision_engine = medichat_vision(user_input, encode_image(uploaded_image), st.session_state.messages, lang_instruction)
                    reply = strip_excessive_disclaimers(reply)
                st.session_state.last_image_context = (
                    "User uploaded image: " + uploaded_image.name + "\n"
                    "User's question about image: " + (user_input.strip() if user_input.strip() else "(no question, just the image)") + "\n"
                    "Your visual analysis: " + reply
                )
                st.session_state.last_sources = ["Image Analysis"]
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply, "sources": st.session_state.last_sources, "confidence": "medium", "confidence_pct": 75, "engine": vision_engine})
                st.session_state.uploader_key += 1
        else:
            user_msg = {"role": "user", "type": "text", "content": user_input.strip()}
            st.session_state.messages.append(user_msg)
            _t0 = time.time()

            name_for_rag = "" if st.session_state.patient_name == "Guest" else st.session_state.patient_name

            with st.spinner("MediChat is thinking..."):
                final_text = ""
                stream_metadata = None
                try:
                    for event in medichat_rag_stream(user_input, st.session_state.messages, lang_instruction, name_for_rag, st.session_state.get("last_pdf_context", ""), st.session_state.get("last_image_context", "")):
                        kind = event[0]
                        if kind == "chunk":
                            final_text = event[2]
                        elif kind == "done":
                            final_text = event[1]
                            stream_metadata = event[2]
                except Exception as e:
                    st.error("MediChat had trouble generating a response. Please try again.")
                    st.stop()

            final_text = strip_excessive_disclaimers(final_text)

            if stream_metadata is None:
                stream_metadata = {"memory": st.session_state.patient_memory, "sources": [], "confidence": "low", "confidence_pct": 0, "engine": "unknown"}

            memory = stream_metadata["memory"]
            sources = stream_metadata["sources"]
            conf_level = stream_metadata["confidence"]
            conf_pct = stream_metadata["confidence_pct"]
            engine_used = stream_metadata.get("engine", "unknown")
            st.session_state.patient_memory = memory
            st.session_state.last_sources = sources
            if st.session_state.is_authenticated and st.session_state.user_email_hash:
                persist_profile_state(
                    st.session_state.user_email_hash,
                    patient_memory=memory,
                    name=st.session_state.patient_name or None,
                    language=st.session_state.selected_language,
                )
            _response_time = round(time.time() - _t0, 2)
            st.session_state.response_times.append(_response_time)

            interaction_alerts = check_drug_interactions(final_text, memory)
            if interaction_alerts:
                alert_block = "\n\n---\n\n**Drug Safety Check:**\n"
                for a in interaction_alerts:
                    alert_block += "\n- **" + a["drug"] + "** — given your " + ", ".join(a["conditions"]) + ": " + a["warning"]
                final_text = final_text + alert_block

            _log_entry = {
                "query": user_input.strip(),
                "confidence": conf_level,
                "confidence_pct": conf_pct,
                "sources": sources,
                "response_time": _response_time,
                "language": st.session_state.selected_language,
                "mode": "free_chat",
                "emergency_triggered": st.session_state.emergency_detected,
                "drug_alerts": len(interaction_alerts),
                "engine": engine_used,
            }
            st.session_state.eval_log.append(_log_entry)
            log_query_to_firestore(_log_entry)

            st.session_state.messages.append({"role": "assistant", "type": "text", "content": final_text, "sources": sources, "confidence": conf_level, "confidence_pct": conf_pct, "engine": engine_used})

        if st.session_state.is_authenticated and st.session_state.user_email_hash:
            persist_profile_state(
                st.session_state.user_email_hash,
                patient_memory=st.session_state.patient_memory,
                name=st.session_state.patient_name or None,
                language=st.session_state.selected_language,
                messages=st.session_state.messages,
            )
        st.session_state.chat_input_key = st.session_state.get("chat_input_key", 0) + 1
        st.rerun()

elif st.session_state.mode == "eval":
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1f2937,#111827);color:white;padding:0.7rem 1.2rem;border-radius:12px;margin-bottom:1rem;display:flex;align-items:center;justify-content:space-between;">'
        '<div style="display:flex;align-items:center;gap:0.5rem;"><span style="font-size:1rem;">🔒</span><span style="font-weight:600;font-size:0.9rem;">Admin Mode — Research & Evaluation Dashboard</span></div>'
        '<div style="font-size:0.75rem;opacity:0.8;">Not visible to patients</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### 📊 MediChat Analytics Dashboard")
    st.caption("Real-time session analytics for clinical evaluation and research reporting. All patient query text is anonymised.")

    dc1, dc2 = st.columns([3, 2])
    with dc2:
        if FIREBASE_ACTIVE:
            data_source = st.radio(
                "Data source:",
                ["Current Session", "All Patients (Firestore)"],
                horizontal=True,
                key="analytics_data_source",
                label_visibility="collapsed"
            )
        else:
            data_source = "Current Session"
            st.caption("Firestore not connected. Showing current session only.")

    if data_source == "All Patients (Firestore)" and FIREBASE_ACTIVE:
        raw_logs = fetch_all_queries_from_firestore(limit=500)
        logs = [
            {
                "query": " " * d.get("query_word_count", 0),
                "confidence": d.get("confidence", "unknown"),
                "confidence_pct": d.get("confidence_pct", 0),
                "sources": d.get("sources", []),
                "response_time": d.get("response_time", 0),
                "language": d.get("language", "English"),
                "emergency_triggered": d.get("emergency_triggered", False),
                "drug_alerts": d.get("drug_alerts", 0),
            }
            for d in raw_logs
        ]
        st.info("Live data from Firestore — aggregated from all MediChat patients (anonymised). Total records: " + str(len(logs)))
    else:
        logs = st.session_state.eval_log
        if FIREBASE_ACTIVE:
            st.caption("Current session only. Switch to 'All Patients (Firestore)' to see aggregate data.")
    total_queries = len(logs)

    if total_queries == 0:
        st.markdown(
            '<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:14px;padding:2rem;text-align:center;color:#0369a1;">'
            '<div style="font-size:3rem;margin-bottom:0.8rem;">📊</div>'
            '<div style="font-size:1.1rem;font-weight:600;margin-bottom:0.4rem;">No data yet</div>'
            '<div style="font-size:0.9rem;">Start a chat or symptom assessment in Free Chat mode. Analytics will appear here automatically.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Queries", total_queries)
        with mc2:
            avg_conf = sum(l["confidence_pct"] for l in logs) / total_queries
            st.metric("Avg Confidence", str(round(avg_conf, 1)) + "%")
        with mc3:
            avg_time = sum(l["response_time"] for l in logs) / total_queries
            st.metric("Avg Response", str(round(avg_time, 2)) + "s")
        with mc4:
            feedback_data = st.session_state.feedback
            helpful_count = sum(1 for v in feedback_data.values() if v == "helpful")
            not_helpful = sum(1 for v in feedback_data.values() if v == "not_helpful")
            feedback_total = helpful_count + not_helpful
            feedback_pct = round((helpful_count / feedback_total * 100), 1) if feedback_total > 0 else "N/A"
            st.metric("Satisfaction", str(feedback_pct) + ("%" if feedback_total > 0 else ""))

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Confidence Distribution")
            high = sum(1 for l in logs if l["confidence"] == "high")
            medium = sum(1 for l in logs if l["confidence"] == "medium")
            low = sum(1 for l in logs if l["confidence"] == "low")
            st.markdown(
                '<div style="padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;">'
                '<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;">'
                '<div style="flex:0 0 80px;font-size:0.82rem;color:#166534;font-weight:600;">High</div>'
                '<div style="flex:1;height:18px;background:#e5e7eb;border-radius:50px;overflow:hidden;">'
                '<div style="height:100%;width:' + str(round((high / total_queries) * 100)) + '%;background:#22c55e;"></div>'
                '</div>'
                '<div style="flex:0 0 50px;font-size:0.82rem;color:#64748b;text-align:right;">' + str(high) + '</div>'
                '</div>'
                '<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;">'
                '<div style="flex:0 0 80px;font-size:0.82rem;color:#92400e;font-weight:600;">Medium</div>'
                '<div style="flex:1;height:18px;background:#e5e7eb;border-radius:50px;overflow:hidden;">'
                '<div style="height:100%;width:' + str(round((medium / total_queries) * 100)) + '%;background:#f59e0b;"></div>'
                '</div>'
                '<div style="flex:0 0 50px;font-size:0.82rem;color:#64748b;text-align:right;">' + str(medium) + '</div>'
                '</div>'
                '<div style="display:flex;align-items:center;gap:0.8rem;">'
                '<div style="flex:0 0 80px;font-size:0.82rem;color:#991b1b;font-weight:600;">Low</div>'
                '<div style="flex:1;height:18px;background:#e5e7eb;border-radius:50px;overflow:hidden;">'
                '<div style="height:100%;width:' + str(round((low / total_queries) * 100)) + '%;background:#ef4444;"></div>'
                '</div>'
                '<div style="flex:0 0 50px;font-size:0.82rem;color:#64748b;text-align:right;">' + str(low) + '</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col_b:
            st.markdown("#### Source Usage (RAG Retrieval)")
            pubmed_queries = sum(1 for l in logs if any("PubMed" in s for s in l.get("sources", [])))
            dialog_queries = sum(1 for l in logs if any("Doctor-Patient" in s for s in l.get("sources", [])))
            both = sum(1 for l in logs if len(l.get("sources", [])) > 1)
            st.markdown(
                '<div style="padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;">'
                '<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;">'
                '<div style="flex:0 0 110px;font-size:0.82rem;color:#0369a1;font-weight:600;">PubMed</div>'
                '<div style="flex:1;height:18px;background:#e5e7eb;border-radius:50px;overflow:hidden;">'
                '<div style="height:100%;width:' + str(round((pubmed_queries / total_queries) * 100)) + '%;background:#0ea5e9;"></div>'
                '</div>'
                '<div style="flex:0 0 50px;font-size:0.82rem;color:#64748b;text-align:right;">' + str(pubmed_queries) + '</div>'
                '</div>'
                '<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;">'
                '<div style="flex:0 0 110px;font-size:0.82rem;color:#7c3aed;font-weight:600;">Doctor-Patient</div>'
                '<div style="flex:1;height:18px;background:#e5e7eb;border-radius:50px;overflow:hidden;">'
                '<div style="height:100%;width:' + str(round((dialog_queries / total_queries) * 100)) + '%;background:#a855f7;"></div>'
                '</div>'
                '<div style="flex:0 0 50px;font-size:0.82rem;color:#64748b;text-align:right;">' + str(dialog_queries) + '</div>'
                '</div>'
                '<div style="display:flex;align-items:center;gap:0.8rem;">'
                '<div style="flex:0 0 110px;font-size:0.82rem;color:#059669;font-weight:600;">Mixed Sources</div>'
                '<div style="flex:1;height:18px;background:#e5e7eb;border-radius:50px;overflow:hidden;">'
                '<div style="height:100%;width:' + str(round((both / total_queries) * 100)) + '%;background:#10b981;"></div>'
                '</div>'
                '<div style="flex:0 0 50px;font-size:0.82rem;color:#64748b;text-align:right;">' + str(both) + '</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown("#### Safety Metrics")
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            emergency_fires = sum(1 for l in logs if l.get("emergency_triggered"))
            st.metric("🚨 Emergency Alerts Fired", emergency_fires)
        with sm2:
            drug_warnings = sum(l.get("drug_alerts", 0) for l in logs)
            st.metric("⚠️ Drug Safety Warnings", drug_warnings)
        with sm3:
            mem = st.session_state.patient_memory
            total_extracted = len(mem.get("symptoms", [])) + len(mem.get("conditions", [])) + len(mem.get("medications", []))
            st.metric("🧠 Memory Entries Extracted", total_extracted)

        st.markdown("---")

        lang_counts = {}
        for l in logs:
            lang = l.get("language", "English")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if len(lang_counts) > 0:
            st.markdown("#### Language Distribution")
            lang_html = '<div style="padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;">'
            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                pct = round((count / total_queries) * 100)
                lang_html += (
                    '<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">'
                    '<div style="flex:0 0 110px;font-size:0.82rem;color:#334155;font-weight:600;">' + lang + '</div>'
                    '<div style="flex:1;height:16px;background:#e5e7eb;border-radius:50px;overflow:hidden;">'
                    '<div style="height:100%;width:' + str(pct) + '%;background:#8b5cf6;"></div>'
                    '</div>'
                    '<div style="flex:0 0 70px;font-size:0.82rem;color:#64748b;text-align:right;">' + str(count) + ' (' + str(pct) + '%)</div>'
                    '</div>'
                )
            lang_html += '</div>'
            st.markdown(lang_html, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("#### Recent Queries Log (Anonymised)")
        st.caption("For patient privacy, query text is not displayed. Only query length, confidence, timing, and source metadata are shown.")
        log_html = '<div style="padding:1rem;background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;max-height:300px;overflow-y:auto;">'
        for i, l in enumerate(reversed(logs[-10:]), 1):
            conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(l["confidence"], "#64748b")
            query_len = len(l["query"].split())
            log_html += (
                '<div style="padding:0.6rem 0.8rem;border-bottom:1px solid #e5e7eb;font-size:0.8rem;">'
                '<div style="color:#334155;margin-bottom:0.2rem;font-weight:600;">Query #' + str(total_queries - i + 1) + ' — ' + str(query_len) + ' words</div>'
                '<div style="display:flex;gap:0.8rem;font-size:0.7rem;color:#64748b;flex-wrap:wrap;">'
                '<span style="color:' + conf_color + ';font-weight:600;">● ' + str(l["confidence_pct"]) + '% conf</span>'
                '<span>⏱ ' + str(l["response_time"]) + 's</span>'
                '<span>📚 ' + ", ".join(l.get("sources", []) or ["—"]) + '</span>'
                '<span>🌐 ' + l.get("language", "English") + '</span>'
                '</div>'
                '</div>'
            )
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

        st.markdown("---")

        ec1, ec2 = st.columns([1, 1])
        with ec1:
            import io as _io
            try:
                from openpyxl import Workbook as _Workbook
                from openpyxl.styles import Font as _Font, PatternFill as _Fill, Alignment as _Align, Border as _Border, Side as _Side
                from openpyxl.utils import get_column_letter as _col_letter

                wb = _Workbook()
                thin = _Side(border_style="thin", color="CCCCCC")
                cell_border = _Border(top=thin, bottom=thin, left=thin, right=thin)
                header_fill = _Fill(start_color="1F3864", end_color="1F3864", fill_type="solid")
                header_font = _Font(name="Arial", bold=True, color="FFFFFF", size=11)
                label_font = _Font(name="Arial", bold=True, size=11, color="1F3864")
                body_font = _Font(name="Arial", size=10)

                ws1 = wb.active
                ws1.title = "Summary"
                ws1["A1"] = "MediChat Analytics Summary"
                ws1["A1"].font = _Font(name="Arial", bold=True, size=16, color="1F3864")
                ws1.merge_cells("A1:B1")
                ws1["A2"] = "Generated: " + datetime.now().strftime("%B %d, %Y at %I:%M %p")
                ws1["A2"].font = _Font(name="Arial", italic=True, size=10, color="555555")
                ws1.merge_cells("A2:B2")
                ws1["A3"] = "Privacy Note: All patient query text has been anonymised."
                ws1["A3"].font = _Font(name="Arial", italic=True, size=9, color="C2410C")
                ws1.merge_cells("A3:B3")

                summary_rows = [
                    ("Metric", "Value"),
                    ("Total Queries", total_queries),
                    ("Average Confidence (%)", round(avg_conf, 2)),
                    ("Average Response Time (seconds)", round(avg_time, 2)),
                    ("Patient Satisfaction (%)", feedback_pct if feedback_total > 0 else "No feedback yet"),
                    ("", ""),
                    ("Confidence Distribution", ""),
                    ("  High Confidence", high),
                    ("  Medium Confidence", medium),
                    ("  Low Confidence", low),
                    ("", ""),
                    ("Source Usage (RAG Retrieval)", ""),
                    ("  PubMed Research", pubmed_queries),
                    ("  Doctor-Patient Data", dialog_queries),
                    ("  Mixed Sources", both),
                    ("", ""),
                    ("Safety Metrics", ""),
                    ("  Emergency Alerts Fired", emergency_fires),
                    ("  Drug Safety Warnings Raised", drug_warnings),
                    ("  Memory Entries Extracted", total_extracted),
                ]
                for r, (label, value) in enumerate(summary_rows, start=5):
                    ws1.cell(row=r, column=1, value=label).font = header_font if r == 5 else label_font
                    ws1.cell(row=r, column=2, value=value).font = header_font if r == 5 else body_font
                    if r == 5:
                        ws1.cell(row=r, column=1).fill = header_fill
                        ws1.cell(row=r, column=2).fill = header_fill
                    ws1.cell(row=r, column=1).border = cell_border
                    ws1.cell(row=r, column=2).border = cell_border
                ws1.column_dimensions["A"].width = 42
                ws1.column_dimensions["B"].width = 22

                ws2 = wb.create_sheet("Languages")
                ws2["A1"] = "Language Distribution"
                ws2["A1"].font = _Font(name="Arial", bold=True, size=14, color="1F3864")
                ws2.merge_cells("A1:C1")
                ws2["A3"] = "Language"
                ws2["B3"] = "Query Count"
                ws2["C3"] = "Percentage"
                for c in ["A3", "B3", "C3"]:
                    ws2[c].font = header_font
                    ws2[c].fill = header_fill
                    ws2[c].border = cell_border
                row_i = 4
                for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = round((count / total_queries) * 100, 1)
                    ws2.cell(row=row_i, column=1, value=lang).font = body_font
                    ws2.cell(row=row_i, column=2, value=count).font = body_font
                    ws2.cell(row=row_i, column=3, value=str(pct) + "%").font = body_font
                    for col in range(1, 4):
                        ws2.cell(row=row_i, column=col).border = cell_border
                    row_i += 1
                ws2.column_dimensions["A"].width = 18
                ws2.column_dimensions["B"].width = 15
                ws2.column_dimensions["C"].width = 15

                ws3 = wb.create_sheet("Query Log (Anonymised)")
                ws3["A1"] = "Anonymised Query Log"
                ws3["A1"].font = _Font(name="Arial", bold=True, size=14, color="1F3864")
                ws3.merge_cells("A1:H1")
                ws3["A2"] = "All patient query text has been removed. Only metadata is included for privacy compliance."
                ws3["A2"].font = _Font(name="Arial", italic=True, size=9, color="C2410C")
                ws3.merge_cells("A2:H2")

                headers3 = ["Query ID", "Word Count", "Confidence", "Confidence %", "Response Time (s)", "Sources", "Language", "Safety Triggered"]
                for col_i, h in enumerate(headers3, start=1):
                    c = ws3.cell(row=4, column=col_i, value=h)
                    c.font = header_font
                    c.fill = header_fill
                    c.border = cell_border
                    c.alignment = _Align(horizontal="center")

                for row_i, l in enumerate(logs, start=5):
                    safety = "Yes" if (l.get("emergency_triggered") or l.get("drug_alerts", 0) > 0) else "No"
                    values = [
                        row_i - 4,
                        len(l["query"].split()),
                        l["confidence"].title(),
                        str(l["confidence_pct"]) + "%",
                        l["response_time"],
                        ", ".join(l.get("sources", [])) or "—",
                        l.get("language", "English"),
                        safety,
                    ]
                    for col_i, v in enumerate(values, start=1):
                        c = ws3.cell(row=row_i, column=col_i, value=v)
                        c.font = body_font
                        c.border = cell_border
                        c.alignment = _Align(horizontal="center") if col_i != 6 else _Align(horizontal="left")
                widths3 = [10, 12, 14, 14, 16, 30, 12, 16]
                for col_i, w in enumerate(widths3, start=1):
                    ws3.column_dimensions[_col_letter(col_i)].width = w

                ws4 = wb.create_sheet("Methodology")
                ws4["A1"] = "MediChat Analytics Methodology"
                ws4["A1"].font = _Font(name="Arial", bold=True, size=14, color="1F3864")
                ws4.merge_cells("A1:B1")

                method_notes = [
                    ("Confidence Score", "Calculated from FAISS L2 distance between query embedding and top-3 retrieved documents. Lower distance = higher confidence. < 0.8 = High, 0.8-1.3 = Medium, > 1.3 = Low."),
                    ("Source Classification", "PubMed Research = documents 0-499 in FAISS index (biomedical research papers). Doctor-Patient Data = documents 500-999 (real clinical conversations)."),
                    ("Emergency Detection", "Hybrid system: direct keyword match on 30+ emergency terms, plus 6 symptom-cluster patterns (cardiac, stroke, anaphylaxis, syncope, severe asthma, hyperglycemic crisis)."),
                    ("Drug Safety Warnings", "Hard-coded rules cross-reference 6 drug classes against patient-stated conditions from memory extraction."),
                    ("Response Time", "Measured from query submission to final response display, including FAISS retrieval and LLM inference."),
                    ("Privacy Compliance", "Patient query text is NEVER stored or exported. Only aggregate metadata and word counts are retained for evaluation purposes."),
                    ("Data Retention", "All analytics data is held in browser session state only. Cleared automatically when the patient closes their tab or clicks Reset."),
                ]
                for r, (label, desc) in enumerate(method_notes, start=3):
                    ws4.cell(row=r, column=1, value=label).font = label_font
                    ws4.cell(row=r, column=1).alignment = _Align(vertical="top")
                    ws4.cell(row=r, column=2, value=desc).font = body_font
                    ws4.cell(row=r, column=2).alignment = _Align(wrap_text=True, vertical="top")
                    ws4.row_dimensions[r].height = 60
                ws4.column_dimensions["A"].width = 24
                ws4.column_dimensions["B"].width = 75

                _buffer = _io.BytesIO()
                wb.save(_buffer)
                _buffer.seek(0)

                st.download_button(
                    "📊 Export Analytics (Excel)",
                    data=_buffer.getvalue(),
                    file_name="medichat_analytics_" + datetime.now().strftime("%Y%m%d_%H%M") + ".xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as _ex:
                st.error("Excel export failed: " + str(_ex))
                st.info("openpyxl library missing. Add 'openpyxl' to requirements.txt and redeploy.")
        with ec2:
            if st.button("🔄 Reset Analytics", use_container_width=True):
                st.session_state.eval_log = []
                st.session_state.response_times = []
                st.rerun()

else:
    if st.session_state.assessment_complete and st.session_state.assessment_parsed:
        parsed = st.session_state.assessment_parsed
        data = st.session_state.assessment_data
        urgency = parsed.get("urgency", "See a doctor soon")
        urgency_lower = urgency.lower()
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        st.markdown("---")
        st.markdown("### " + L["report_title"])
        st.caption("Generated: " + report_date)

        if "emergency" in urgency_lower or "now" in urgency_lower:
            st.error("URGENCY: " + urgency)
        elif "urgent" in urgency_lower or "today" in urgency_lower:
            st.warning("URGENCY: " + urgency)
        else:
            st.success("URGENCY: " + urgency)

        st.markdown("---")
        st.markdown("#### " + L["symptoms_reported"])
        rc1, rc2 = st.columns(2)
        with rc1:
            st.info("**Main symptom:** " + data.get("main_symptom", ""))
            st.info("**Duration:** " + data.get("duration", ""))
            st.info("**Severity:** " + data.get("severity", ""))
        with rc2:
            st.info("**Pattern:** " + data.get("pattern", ""))
            st.info("**Age:** " + data.get("age", ""))
            st.info("**Sex:** " + data.get("gender", ""))

        other = data.get("other_symptoms", "")
        if other and other.lower() not in ["no", "none", "n/a", "no other symptoms"]:
            st.info("**Other symptoms:** " + other)

        st.markdown("---")
        st.markdown("#### " + L["possible_conditions"])
        for c in parsed.get("conditions", []):
            if c.strip():
                st.markdown("- " + c.strip())

        st.markdown("---")
        st.markdown("#### " + L["what_to_do"])
        for i, s in enumerate(parsed.get("next_steps", []), 1):
            if s.strip():
                st.markdown("**" + str(i) + ".** " + s.strip())

        st.markdown("---")
        st.markdown("#### " + L["summary"])
        summary = parsed.get("summary", "")
        if summary:
            st.markdown('<div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:12px;padding:1rem;font-size:0.92rem;color:#134e4a;line-height:1.6;">' + summary + "</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.warning(L["disclaimer_short"])

        st.markdown("---")
        st.markdown("**" + L["download_chat"] + "**")
        pdf_bytes = generate_assessment_pdf(parsed, data, report_date)
        st.download_button(label=L["download_assess_btn"], data=pdf_bytes, file_name="MediChat_Assessment_" + datetime.now().strftime("%Y%m%d_%H%M") + ".pdf", mime="application/pdf", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        br1, br2 = st.columns(2)
        with br1:
            if st.button(L["new_assessment"], use_container_width=True):
                st.session_state.assessment_stage = 0
                st.session_state.assessment_data = {}
                st.session_state.assessment_complete = False
                st.session_state.assessment_report = None
                st.session_state.assessment_parsed = None
                st.rerun()
        with br2:
            if st.button(L["switch_chat"], use_container_width=True):
                st.session_state.mode = "chat"
                st.rerun()

    else:
        L = LANGUAGES[st.session_state.selected_language]
        stage = st.session_state.assessment_stage
        total = len(ASSESSMENT_STAGES)
        progress = int((stage / total) * 100)

        st.markdown('<div class="assessment-card"><div class="assessment-title">' + L["symptom_title"] + '</div><div class="assessment-subtitle">' + L["symptom_subtitle"] + '</div><div class="progress-label"><span>Step ' + str(stage + 1) + ' of ' + str(total) + '</span><span>' + str(progress) + '% complete</span></div><div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:' + str(progress) + '%;"></div></div></div>', unsafe_allow_html=True)

        if st.session_state.assessment_data:
            with st.expander(L["answers_so_far"], expanded=False):
                for k, v in st.session_state.assessment_data.items():
                    st.markdown("**" + k.replace("_", " ").title() + ":** " + str(v))

        if stage < total:
            current = ASSESSMENT_STAGES[stage]
            st.markdown('<div class="question-bubble">' + current["question"] + '</div>', unsafe_allow_html=True)
            if current["hint"]:
                st.caption(current["hint"])

            if current["options"]:
                st.markdown("**" + L["quick_select"] + "**")
                num_cols = min(len(current["options"]), 3)
                ocols = st.columns(num_cols)
                for i, opt in enumerate(current["options"]):
                    with ocols[i % num_cols]:
                        if st.button(opt, key="opt_" + str(stage) + "_" + str(i), use_container_width=True):
                            st.session_state.assessment_data[current["key"]] = opt
                            if current["key"] == "main_symptom" and detect_emergency(opt)[0]:
                                st.session_state.emergency_detected = True
                            st.session_state.assessment_stage += 1
                            if st.session_state.assessment_stage >= total:
                                lang_instruction = LANGUAGES[st.session_state.selected_language]["lang_instruction"]
                                with st.spinner("Generating your personalised assessment..."):
                                    report = generate_assessment_report(st.session_state.assessment_data, lang_instruction)
                                    st.session_state.assessment_report = report
                                    st.session_state.assessment_parsed = parse_report(report)
                                    st.session_state.assessment_complete = True
                            st.rerun()

            with st.form(key="assessment_form_" + str(stage), clear_on_submit=True):
                typed = st.text_input("", placeholder="Or type your own answer here...", label_visibility="collapsed")
                ac1, ac2, ac3 = st.columns([2, 2, 1])
                with ac2:
                    next_btn = st.form_submit_button(L["next"])
                with ac3:
                    cancel_btn = st.form_submit_button(L["cancel"])

            if next_btn and typed.strip():
                st.session_state.assessment_data[current["key"]] = typed.strip()
                if current["key"] == "main_symptom" and detect_emergency(typed)[0]:
                    st.session_state.emergency_detected = True
                st.session_state.assessment_stage += 1
                if st.session_state.assessment_stage >= total:
                    lang_instruction = LANGUAGES[st.session_state.selected_language]["lang_instruction"]
                    with st.spinner("Generating your personalised assessment..."):
                        report = generate_assessment_report(st.session_state.assessment_data, lang_instruction)
                        st.session_state.assessment_report = report
                        st.session_state.assessment_parsed = parse_report(report)
                        st.session_state.assessment_complete = True
                st.rerun()

            if cancel_btn:
                st.session_state.assessment_stage = 0
                st.session_state.assessment_data = {}
                st.session_state.mode = "chat"
                st.rerun()
