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
from PIL import Image
import io
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

def log_query_to_firestore(query_data):
    """Write anonymised query metadata to Firestore. Silent failure if Firebase unavailable."""
    if not FIREBASE_ACTIVE:
        return
    try:
        # Strip raw query text before writing - ONLY metadata
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

    /* ═══════════════════════════════════════════════════════════════════
       MediChat Trust-First Design System
       Palette: Sage primary (calm medical), warm ivory canvas, soft accents
    ═══════════════════════════════════════════════════════════════════ */

    :root {
        --sage-900: #1e3a36;
        --sage-700: #2d5a52;
        --sage-500: #5d8b7c;
        --sage-300: #a8c5bd;
        --sage-100: #e8f0ed;
        --sage-50: #f3f8f6;

        --ivory: #fefdfb;
        --cream: #faf8f3;
        --warm-gray: #6b6660;
        --soft-gray: #94908a;

        --accent-rose: #c4766a;
        --accent-amber: #d69e2e;
        --accent-lavender: #8b7aa8;
    }

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(ellipse 80% 60% at top left, rgba(168, 197, 189, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse 70% 50% at bottom right, rgba(139, 122, 168, 0.08) 0%, transparent 50%),
            linear-gradient(180deg, #fefdfb 0%, #faf8f3 100%);
        min-height: 100vh;
    }

    .main .block-container {
        padding: 1rem 1.5rem 2rem 1.5rem;
        max-width: 760px;
    }

    /* ── Header ──────────────────────────────────────────────────── */
    .header-card {
        background: white;
        border-radius: 20px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 0.8rem;
        box-shadow:
            0 1px 3px rgba(30, 58, 54, 0.04),
            0 8px 28px rgba(30, 58, 54, 0.05);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(168, 197, 189, 0.2);
    }

    .header-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--sage-500), var(--accent-lavender), var(--sage-500));
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
        gap: 0.9rem;
        margin-bottom: 0;
    }

    .header-logo {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--sage-500), var(--sage-700));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        color: white;
        box-shadow: 0 3px 12px rgba(93, 139, 124, 0.22);
        flex-shrink: 0;
    }

    .header-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.75rem;
        font-weight: 400;
        color: var(--sage-900);
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1;
    }

    .header-subtitle {
        color: var(--warm-gray);
        font-size: 0.8rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
        line-height: 1.4;
    }

    /* ── Trust Strip ─────────────────────────────────────────────── */
    .trust-strip {
        display: flex;
        gap: 0.55rem;
        margin: 1rem 0 1.2rem 0;
        flex-wrap: wrap;
        justify-content: center;
    }

    .trust-pill {
        background: white;
        border: 1px solid var(--sage-100);
        border-radius: 100px;
        padding: 0.4rem 0.9rem;
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--sage-700);
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 1px 2px rgba(30, 58, 54, 0.03);
        transition: all 0.2s ease;
    }

    .trust-pill:hover {
        border-color: var(--sage-300);
        transform: translateY(-1px);
    }

    .trust-pill-icon {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: var(--sage-500);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.55rem;
        font-weight: 700;
    }

    /* Legacy stats-row classes kept for compatibility */
    .stats-row { display: flex; gap: 0.55rem; margin-bottom: 1rem; flex-wrap: wrap; justify-content: center; }
    .stat-pill { background: white; border: 1px solid var(--sage-100); border-radius: 100px; padding: 0.4rem 0.9rem; font-size: 0.72rem; font-weight: 500; color: var(--sage-700); }
    .stat-pill.green { color: var(--sage-700); border-color: var(--sage-300); background: var(--sage-50); }
    .stat-pill.blue { color: #3a6b8f; border-color: #bed2e0; background: #eff5fa; }
    .stat-pill.purple { color: var(--accent-lavender); border-color: #d4c9e3; background: #f5f0fa; }
    .stat-pill.orange { color: var(--accent-amber); border-color: #e8cf9e; background: #fbf5e7; }

    /* ── Disclaimer ──────────────────────────────────────────────── */
    .disclaimer {
        display: none;
    }

    .disclaimer-mini {
        font-size: 0.7rem;
        color: var(--soft-gray);
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
        border-radius: 16px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.25);
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 8px 24px rgba(220, 38, 38, 0.25); }
        50% { box-shadow: 0 8px 36px rgba(220, 38, 38, 0.55); }
    }
    .emergency-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem; }
    .emergency-text { font-size: 0.85rem; line-height: 1.55; margin-bottom: 0.6rem; opacity: 0.95; }
    .emergency-number {
        background: white; color: #991b1b;
        padding: 0.55rem 1.2rem; border-radius: 12px;
        font-size: 1.15rem; font-weight: 700;
        display: inline-block; margin-top: 0.2rem;
        letter-spacing: 0.04em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* ── Mode Buttons ────────────────────────────────────────────── */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 14px !important;
        border: 1px solid var(--sage-100) !important;
        background: white !important;
        color: var(--sage-700) !important;
        padding: 0.7rem 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(30, 58, 54, 0.04) !important;
    }
    .stButton > button:hover {
        border-color: var(--sage-300) !important;
        background: var(--sage-50) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(93, 139, 124, 0.12) !important;
    }

    /* ── Welcome Card ────────────────────────────────────────────── */
    .welcome-card {
        background: white;
        border-radius: 22px;
        padding: 2.4rem 2.2rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(30, 58, 54, 0.06);
        margin: 0.5rem 0 1.2rem 0;
        border: 1px solid rgba(168, 197, 189, 0.2);
        animation: fadeInUp 0.5s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .welcome-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.7rem;
        font-weight: 400;
        color: var(--sage-900);
        margin: 0.8rem 0 0.6rem 0;
        letter-spacing: -0.01em;
    }
    .welcome-text {
        color: var(--warm-gray);
        font-size: 0.95rem;
        line-height: 1.65;
        margin-bottom: 1.4rem;
    }

    .chip-row {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 1rem;
    }
    .chip {
        background: var(--sage-50);
        border: 1px solid var(--sage-100);
        color: var(--sage-700);
        padding: 0.4rem 0.9rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .chip:hover {
        background: var(--sage-100);
        border-color: var(--sage-300);
        transform: translateY(-1px);
    }

    /* ── Memory Card ─────────────────────────────────────────────── */
    .memory-card {
        background: linear-gradient(135deg, #f3f8f6, #eef4f1);
        border: 1px solid var(--sage-100);
        border-radius: 14px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.8rem;
        color: var(--sage-700);
        animation: fadeIn 0.4s ease;
    }
    .memory-title { font-weight: 600; margin-bottom: 0.35rem; font-size: 0.82rem; color: var(--sage-900); display: flex; align-items: center; gap: 0.4rem; }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* ── Chat Messages ───────────────────────────────────────────── */
    .bot-label {
        font-size: 0.68rem;
        color: var(--soft-gray);
        font-weight: 600;
        margin-left: 54px;
        margin-bottom: 0.3rem;
        margin-top: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .bot-wrap, .user-wrap {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        margin-bottom: 0.8rem;
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
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.95rem;
        flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(30, 58, 54, 0.1);
    }

    .av-bot {
        background: linear-gradient(135deg, var(--sage-500), var(--sage-700));
        color: white;
    }

    .av-user {
        background: linear-gradient(135deg, #e8e5df, #d6d0c5);
        color: var(--sage-900);
    }

    .bot-bubble {
        background: white;
        color: #2a2825;
        padding: 1rem 1.25rem;
        border-radius: 4px 18px 18px 18px;
        max-width: 85%;
        font-size: 0.93rem;
        line-height: 1.65;
        box-shadow: 0 1px 3px rgba(30, 58, 54, 0.04), 0 8px 24px rgba(30, 58, 54, 0.04);
        border: 1px solid rgba(168, 197, 189, 0.15);
        position: relative;
    }

    .bot-bubble::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--sage-500), var(--sage-300));
        border-radius: 4px 0 0 0;
    }

    .user-bubble {
        background: linear-gradient(135deg, var(--sage-700), var(--sage-900));
        color: white;
        padding: 0.85rem 1.15rem;
        border-radius: 18px 4px 18px 18px;
        max-width: 80%;
        font-size: 0.93rem;
        line-height: 1.55;
        box-shadow: 0 4px 12px rgba(30, 58, 54, 0.15);
    }

    /* ── Thinking Indicator ──────────────────────────────────────── */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.8rem 1rem;
        background: white;
        border-radius: 4px 18px 18px 18px;
        border-left: 3px solid var(--sage-500);
        max-width: 180px;
        box-shadow: 0 1px 3px rgba(30, 58, 54, 0.04);
    }
    .thinking-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--sage-500);
        animation: bounce 1.4s infinite ease-in-out;
    }
    .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }
    .thinking-label {
        font-size: 0.78rem;
        color: var(--soft-gray);
        margin-left: 0.3rem;
        font-style: italic;
    }

    /* ── Streaming Cursor ────────────────────────────────────────── */
    .stream-cursor {
        display: inline-block;
        width: 2px;
        height: 1em;
        background: var(--sage-500);
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
        margin-left: 54px;
        margin-bottom: 0.3rem;
        font-size: 0.72rem;
        color: var(--soft-gray);
    }

    .engine-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 100px;
        letter-spacing: 0.02em;
        margin-right: 0.4rem;
    }
    .engine-claude {
        background: linear-gradient(135deg, #fef2e8, #fdeede);
        color: #a8521a;
        border: 1px solid #f5c4a1;
    }
    .engine-groq {
        background: var(--sage-50);
        color: var(--sage-700);
        border: 1px solid var(--sage-300);
    }
    .engine-badge::before {
        content: "●";
        font-size: 0.5rem;
    }
    .source-tag {
        display: inline-block;
        background: var(--sage-50);
        border: 1px solid var(--sage-100);
        color: var(--sage-700);
        font-size: 0.68rem;
        font-weight: 500;
        padding: 0.2rem 0.65rem;
        border-radius: 100px;
        margin-right: 0.3rem;
        margin-top: 0.3rem;
    }
    .confidence-row {
        margin-left: 54px;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.68rem;
    }
    .confidence-pill {
        padding: 0.15rem 0.7rem;
        border-radius: 100px;
        font-weight: 600;
        letter-spacing: 0.03em;
        font-size: 0.66rem;
    }
    .conf-high { background: var(--sage-50); color: var(--sage-700); border: 1px solid var(--sage-300); }
    .conf-medium { background: #fbf5e7; color: #7a5d1a; border: 1px solid #e8cf9e; }
    .conf-low { background: #fdf2f1; color: #8f3f34; border: 1px solid #e3bfb8; }
    .confidence-bar {
        display: inline-block;
        width: 80px;
        height: 5px;
        background: var(--sage-100);
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
        border-radius: 14px !important;
        border: 1.5px solid var(--sage-100) !important;
        padding: 0.8rem 1.1rem !important;
        font-size: 0.93rem !important;
        background: white !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--sage-500) !important;
        box-shadow: 0 0 0 3px rgba(93, 139, 124, 0.15) !important;
    }

    /* ── Image Tag ───────────────────────────────────────────────── */
    .image-tag {
        display: inline-block;
        background: #f5f0fa;
        color: var(--accent-lavender);
        border: 1px solid #d4c9e3;
        padding: 0.3rem 0.8rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    /* ── Name Welcome ────────────────────────────────────────────── */
    .name-welcome {
        background: linear-gradient(135deg, #f3f8f6, #eef4f1);
        border: 1px solid var(--sage-100);
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1rem;
        animation: fadeInUp 0.4s ease;
    }
    .name-welcome-text {
        font-size: 0.92rem;
        color: var(--sage-900);
        font-weight: 400;
    }

    /* ── Suggested Follow-ups ────────────────────────────────────── */
    .suggestion-row {
        margin-left: 54px;
        margin-bottom: 1rem;
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
    }
    .suggestion-label {
        font-size: 0.68rem;
        color: var(--soft-gray);
        margin-left: 54px;
        margin-bottom: 0.4rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Sidebar Styling ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 1px solid var(--sage-100);
    }
    [data-testid="stSidebar"] .stMarkdown { color: var(--sage-900); }
    .sb-title {
        font-size: 0.67rem;
        font-weight: 700;
        color: var(--soft-gray);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 0.8rem 0 0.5rem 0;
    }
    .sb-stat-card { background: var(--sage-50); border: 1px solid var(--sage-100); border-radius: 12px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; }
    .sb-stat-num { font-size: 1.4rem; font-weight: 700; color: var(--sage-700) !important; line-height: 1; font-family: 'DM Serif Display', serif; }
    .sb-stat-label { font-size: 0.65rem; color: var(--soft-gray) !important; font-weight: 500; margin-top: 0.15rem; }
    .sb-feature { display: flex; align-items: center; gap: 0.55rem; background: var(--sage-50); border: 1px solid var(--sage-100); border-radius: 10px; padding: 0.45rem 0.7rem; margin-bottom: 0.35rem; }
    .sb-feature-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .sb-feature-name { font-size: 0.74rem; font-weight: 500; color: var(--sage-900) !important; }
    .sb-feature-status { font-size: 0.63rem; color: var(--sage-500) !important; margin-left: auto; font-weight: 600; }
    .sb-tip { font-size: 0.73rem; color: var(--warm-gray) !important; padding: 0.3rem 0; border-bottom: 1px solid var(--sage-50); line-height: 1.5; }
    .sb-memory-item { font-size: 0.7rem; color: var(--sage-700) !important; padding: 0.25rem 0; border-bottom: 1px solid var(--sage-50); }
    .sb-footer { font-size: 0.65rem; color: var(--soft-gray) !important; text-align: center; padding-top: 1rem; border-top: 1px solid var(--sage-50); line-height: 1.6; }

    /* ── Hide Streamlit Branding ─────────────────────────────────── */
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }

    /* Hide Streamlit's "Press Enter to submit form" hint that overlaps input */
    [data-testid="InputInstructions"] { display: none !important; }
    .stForm [data-testid="stFormSubmitButton"] + small { display: none !important; }
    .stForm small { display: none !important; }

    /* File uploader error message: place BELOW the file widget instead of overlapping */
    [data-testid="stFileUploader"] [data-testid="stAlert"] {
        margin-top: 0.5rem !important;
        position: relative !important;
        z-index: 1 !important;
    }
    [data-testid="stFileUploader"] section {
        position: relative !important;
        z-index: 2 !important;
    }
    /* Make the X button on uploaded files always clickable */
    [data-testid="stFileUploader"] button[kind="header"] {
        z-index: 10 !important;
        position: relative !important;
    }

</style>
""", unsafe_allow_html=True)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("API key not found.")
    st.stop()
groq_client = Groq(api_key=GROQ_API_KEY)

# Anthropic Claude (primary) with Groq fallback
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

    # Load PubMedQA (primary dataset — 500 biomedical research Q&A)
    try:
        pubmed = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:500]")
        pubmed_docs = ["[PubMed Research]\nQuestion: " + i["question"] + "\nAnswer: " + i["long_answer"] for i in pubmed]
    except Exception as e:
        print("PubMedQA load failed:", e)

    # Load MedDialog (doctor-patient conversations) — try multiple sources
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

    # If both datasets failed, use a minimal built-in fallback so app still works
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
    """Extract text from an uploaded PDF file. Returns text content or empty string on failure."""
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
        for page in reader.pages[:20]:  # cap at 20 pages
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
        full_text = "\n".join(text_parts).strip()
        # Truncate if extremely long
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "\n\n[Document truncated for length]"
        return full_text
    except Exception as e:
        print("PDF extraction failed:", e)
        return ""

def medichat_pdf_analysis(question, pdf_text, all_messages, lang_instruction=""):
    """Analyze an uploaded PDF (e.g., blood test report) using Claude or Groq."""
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

# ── Advanced Emergency Detection ──────────────────────────────────────
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

# Symptom clusters that together indicate emergency (even if individual words are mild)
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
    """Detect emergency via keywords OR symptom cluster pattern matching."""
    if not text and not conversation_text:
        return False, None
    combined = (text + " " + conversation_text).lower()

    # Direct keyword match
    for kw in EMERGENCY_KEYWORDS:
        if kw in combined:
            return True, "Emergency keyword detected"

    # Symptom cluster match
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

# ── Drug-Condition Interaction Safety ─────────────────────────────────
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
    """Scan MediChat's response for drug mentions and cross-check with stated patient conditions."""
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

# ── Response Confidence Scoring ───────────────────────────────────────
def calculate_confidence(distances):
    """Convert FAISS L2 distances into a simple confidence indicator."""
    if not distances or len(distances) == 0:
        return "low", 0
    avg_dist = sum(distances) / len(distances)
    # FAISS L2 with MiniLM: ~0.5 is close match, ~1.5 is moderate, >2 is weak
    if avg_dist < 0.8:
        return "high", round((1 - avg_dist / 2) * 100)
    elif avg_dist < 1.3:
        return "medium", round((1 - avg_dist / 2) * 100)
    else:
        return "low", max(20, round((1 - avg_dist / 2.5) * 100))

# ── Source Tracking ───────────────────────────────────────────────────
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

        "CLINICAL REASONING FRAMEWORK (apply to EVERY condition, not just asthma):\n\n"

        "STEP 1 - ANCHOR ON STATED CONDITIONS:\n"
        "If the patient has already told you they have a diagnosed condition "
        "(asthma, diabetes, hypertension, thyroid, PCOS, migraine, anxiety, IBS, etc.), "
        "make that your PRIMARY lens. New symptoms in a known condition usually point to "
        "either (a) the condition being poorly controlled, (b) a side effect of their medication, "
        "or (c) a common comorbidity. Explore THOSE first before pivoting to unrelated diagnoses.\n\n"

        "STEP 2 - INTEGRATE THE FULL SYMPTOM PICTURE:\n"
        "Do NOT treat symptoms as separate items on a list. Ask: what SINGLE mechanism could "
        "explain all of them together? For example:\n"
        "- Diabetic with fatigue + thirst + blurred vision = uncontrolled blood sugar\n"
        "- Hypertensive with headache + chest pressure + vision changes = hypertensive crisis\n"
        "- Asthmatic with shortness of breath + racing heart + nausea = asthma exacerbation OR beta-agonist side effect\n"
        "- Thyroid patient with tremor + weight loss + anxiety = medication dose too high\n"
        "The unifying diagnosis is almost always more useful than five disconnected possibilities.\n\n"

        "STEP 3 - CHECK MEDICATION-CONDITION SAFETY:\n"
        "Before suggesting ANY medication (OTC or otherwise), mentally check it against the "
        "patient's stated conditions. Flag interactions directly. Examples:\n"
        "- Antihistamines (Dramamine, Bonine, Benadryl) → caution in asthma, glaucoma, BPH\n"
        "- NSAIDs (ibuprofen, naproxen) → caution in hypertension, kidney disease, ulcers, asthma\n"
        "- Decongestants (pseudoephedrine) → caution in hypertension, heart disease, thyroid\n"
        "- Pepto-Bismol → caution in aspirin allergy, kidney disease\n"
        "- Paracetamol → caution in liver disease, heavy alcohol use\n"
        "- PPIs/H2 blockers → check interactions with other meds\n"
        "If you suggest a medication, ALWAYS say 'but with [condition], be cautious because...' "
        "or recommend a safer alternative for their specific situation.\n\n"

        "STEP 4 - ASK THE RIGHT FOLLOW-UP, NOT GENERIC QUESTIONS:\n"
        "Ask ONE targeted clinical question that matches the stated condition. Examples:\n"
        "- Asthma patient: 'Are you using a preventer inhaler daily, or only a reliever when symptoms hit?'\n"
        "- Diabetic: 'What has your blood sugar been reading recently?'\n"
        "- Hypertensive: 'What are your recent blood pressure readings?'\n"
        "- Migraine: 'Have your triggers or frequency changed recently?'\n"
        "Don't ask 'when did symptoms start' unless you genuinely don't have the info.\n\n"

        "STEP 5 - COMMIT TO THE MOST LIKELY DIAGNOSIS:\n"
        "After gathering enough info, state the most likely explanation directly. "
        "Don't hedge with 'it could be many things' — pick the 1-2 most probable causes "
        "based on the full picture and explain WHY. Then list what to ask the doctor for specifically "
        "(tests, referrals, medication reviews).\n\n"

        "STRICT OUTPUT RULES:\n"
        "- MAXIMUM ONE disclaimer at the end of the whole response (not per paragraph). "
        "The app already shows a permanent disclaimer banner.\n"
        "- Do NOT say 'I'm not a doctor' more than once per conversation.\n"
        "- Do NOT repeat the patient's symptoms back to them — they already know what they told you.\n"
        "- Do NOT use excessive empathy filler like 'that sounds overwhelming' or 'that must be really scary'. "
        "A single acknowledgement is fine, not in every message.\n"
        "- Be warm but CONFIDENT. You help patients MORE by giving real answers than by being evasive.\n"
        "- Never invent symptoms or conditions the patient did not state.\n\n"
    )

    if patient_name:
        system += "The patient's name is " + patient_name + ". Use their name sparingly, maximum once per response.\n\n"
    if lang_instruction:
        system += lang_instruction + "\n\n"
    if memory_context:
        system += (
            "WHAT THIS PATIENT HAS TOLD YOU ALREADY (ANCHOR ON THIS):\n"
            + memory_context + "\n\n"
        )

    system += (
        "MEDICAL KNOWLEDGE CONTEXT (from PubMed and real doctor-patient conversations):\n"
        + context
    )

    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}]
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        temperature=0.4,
        max_tokens=1024
    )
    return r.choices[0].message.content, memory, sources, confidence_level, confidence_pct

def sanitize_rag_context(raw_context):
    """
    Strip specific patient details from retrieved RAG documents so the LLM cannot
    mistake them for the current patient's history. Removes names, specific medication
    lists, and first-person narrative fragments that look like patient memory.
    """
    import re as _re
    if not raw_context:
        return ""

    # Medications that commonly leak from MedDialog into retrieved docs
    leaked_meds = [
        "subutex", "neurontin", "gabapentin", "remeron", "mirtazapine",
        "zoloft", "sertraline", "klonopin", "clonazepam", "synthroid",
        "levothyroxine", "xanax", "prozac", "lexapro", "wellbutrin",
        "lisinopril", "metformin", "atorvastatin", "amlodipine",
    ]

    text = raw_context
    # Remove first-person fragments that look like patient narrative
    # e.g. "I take subutex" or "I am on 5 meds"
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
        text = _re.sub(pat, "", text)

    # Remove sentences that name specific leaked medications
    sentences = _re.split(r'(?<=[.!?])\s+', text)
    clean = []
    for s in sentences:
        s_lower = s.lower()
        # Drop sentences that mention specific leaked meds (they belong to other patients)
        if any(m in s_lower for m in leaked_meds):
            continue
        clean.append(s)
    result = " ".join(clean).strip()
    return result if len(result) > 100 else raw_context  # fallback if we stripped too much

def strip_excessive_disclaimers(text):
    """Remove inline disclaimer spam AND em-dashes from LLM output. The app shows ONE mini disclaimer permanently."""
    if not text:
        return text
    import re as _re

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
        cleaned = _re.sub(pat, "", cleaned, flags=_re.IGNORECASE)

    # Replace em-dashes and en-dashes with appropriate punctuation
    # " — " (with spaces) becomes ", " or ". " depending on context
    cleaned = _re.sub(r"\s+[\u2014\u2013]\s+", ", ", cleaned)  # spaced em/en dash → comma
    cleaned = cleaned.replace("\u2014", ", ")  # remaining em-dashes
    cleaned = cleaned.replace("\u2013", ", ")  # remaining en-dashes

    # Clean up duplicate commas/spaces created by replacement
    cleaned = _re.sub(r",\s*,", ",", cleaned)
    cleaned = _re.sub(r"\s{2,}", " ", cleaned)
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned

def medichat_rag_stream(question, all_messages, lang_instruction="", patient_name=""):
    """Streaming version: yields text chunks as they arrive. Returns final metadata via final yield."""
    emb = embedder.encode([question]).astype("float32")
    distances, idxs = index.search(emb, k=3)
    raw_context = "\n\n---\n\n".join([documents[i] for i in idxs[0]])
    clean_context = sanitize_rag_context(raw_context)
    sources = get_sources_used(idxs[0])
    confidence_level, confidence_pct = calculate_confidence(distances[0].tolist())
    memory = extract_patient_memory(all_messages)
    memory_context = build_memory_context(memory)

    # Build conversation history from ACTUAL patient messages only
    history = []
    for m in all_messages[-12:]:
        if m.get("type") == "text":
            history.append({"role": m["role"], "content": m["content"]})

    # Detect dissatisfaction / escalation need
    last_user_message = question.lower() if question else ""
    dissatisfaction_signals = [
        "not helping", "isn't helping", "not useful", "doesn't help",
        "you are not helping", "keep repeating", "already said", "same thing",
        "useless", "unhelpful", "rude", "cold",
    ]
    escalation_needed = any(sig in last_user_message for sig in dissatisfaction_signals)

    # Detect tone complaint
    tone_complaint = any(w in last_user_message for w in ["rude", "cold", "robotic", "unfriendly"])

    system = (
        "You are MediChat — a warm, thoughtful AI health companion. "
        "You care about the person in front of you. You speak like a caring GP who happens to also be a good friend: "
        "kind, genuinely interested, never robotic, never preachy.\n\n"

        "═══════════════════════════════════════════════════════════\n"
        "HARD RULES (NEVER BREAK THESE)\n"
        "═══════════════════════════════════════════════════════════\n\n"

        "RULE 1 — NEVER FABRICATE PATIENT HISTORY:\n"
        "The <patient_history> block below shows EXACTLY what this patient has told you. "
        "The <reference_knowledge> block is generic medical information — it does NOT describe this patient. "
        "You must NEVER say 'you mentioned', 'you said', 'you told me', or 'you're taking' unless "
        "the thing you're referencing appears LITERALLY in <patient_history> or in the conversation turns below. "
        "If you invent medications, conditions, or history the patient didn't state, that is a serious safety failure.\n\n"

        "RULE 2 — RED FLAG SCREENING FIRST:\n"
        "For these presentations, screen for danger signs BEFORE general advice:\n"
        "• Sudden/severe headache → thunderclap onset, neck stiffness, fever, vision changes, weakness, confusion\n"
        "• Chest pain → radiation, sweating, SOB, nausea\n"
        "• Sudden SOB → chest pain, leg swelling, recent surgery/travel\n"
        "• Severe abdominal pain → rigidity, fever, vomiting blood\n"
        "• Neuro symptoms → urgent stroke screen\n"
        "If red flags present: advise emergency care clearly and without hedging.\n\n"

        "RULE 3 — NO REPETITION:\n"
        "Look at your own previous messages in this conversation. Never repeat the same advice twice. "
        "Each response must add something new.\n\n"

        "RULE 4 — ONE DISCLAIMER MAX:\n"
        "The app already shows a disclaimer. Only add 'consult a doctor' phrasing when it's genuinely the most important thing to say "
        "(e.g., red flags or specific prescription drug queries). Do NOT end every response with a disclaimer.\n\n"

        "RULE 5 — NO EM-DASHES OR EN-DASHES:\n"
        "Never use em-dashes (\u2014) or en-dashes (\u2013) in your responses. Use commas, semicolons, colons, periods, "
        "or rewrite the sentence. Em-dashes make text feel AI-generated and stiff. "
        "Example: write 'I hear you, that sounds tough' NOT 'I hear you \u2014 that sounds tough'.\n\n"

        "═══════════════════════════════════════════════════════════\n"
        "TONE & STYLE — read this carefully\n"
        "═══════════════════════════════════════════════════════════\n\n"

        "You are WARM. Not clinical-robotic. Not corporate-bland. A real caring presence.\n\n"
        "Good tone examples:\n"
        "✓ 'That sounds uncomfortable — let's figure out what's going on.'\n"
        "✓ 'Headaches can have so many causes, so bear with me for one or two questions.'\n"
        "✓ 'Before we dig in, a quick check: is this the worst headache you've ever had, or similar to ones you've had before?'\n\n"
        "Bad tone examples (DO NOT do these):\n"
        "✗ 'I'm going to take a focused approach.' (sounds like a robot)\n"
        "✗ 'I'll integrate your symptom into a single mechanism.' (medical-jargon weird)\n"
        "✗ 'Please note that I'm not a doctor, but I'll do my best.' (disclaimer fatigue)\n"
        "✗ Listing 'possible causes: stress, hormones, environment' without commitment (wishy-washy)\n\n"

        "Speak like you're talking to a friend who's not feeling well. Be curious, not procedural. "
        "Don't announce what you're about to do — just do it conversationally.\n\n"

        "═══════════════════════════════════════════════════════════\n"
        "CLINICAL APPROACH (apply invisibly, don't narrate it)\n"
        "═══════════════════════════════════════════════════════════\n\n"
        "1. Anchor on what the patient has stated (conditions, meds, symptoms).\n"
        "2. Integrate symptoms into the simplest unified explanation.\n"
        "3. Check medication safety: antihistamines + asthma, NSAIDs + HTN/ulcers/asthma, decongestants + HTN/thyroid, "
        "paracetamol + liver/alcohol, triptans + cardiovascular disease.\n"
        "4. Ask ONE targeted question that actually advances the diagnosis.\n"
        "5. When you have enough info, commit to the most likely 1-2 causes and give specific, actionable next steps "
        "with exact dosing (e.g., 'ibuprofen 400mg every 6 hours with food, up to 1200mg/day for short-term use').\n\n"

        "═══════════════════════════════════════════════════════════\n"
        "IF THE PATIENT SEEMS FRUSTRATED\n"
        "═══════════════════════════════════════════════════════════\n\n"
        "Acknowledge briefly and genuinely ('I hear you, let me be more direct.'). "
        "Then commit to a specific likely diagnosis. Give concrete, specific interventions they haven't heard yet. "
        "Do NOT repeat hydration/rest/cold compress tips if you've said them before.\n\n"
    )

    if tone_complaint:
        system += (
            "⚠ TONE FEEDBACK DETECTED:\n"
            "The patient has told you your tone felt off (rude, cold, robotic). "
            "Apologize briefly and genuinely in ONE short sentence, then show warmth — "
            "be gentler, more conversational, more human. No clinical framework language in this response.\n\n"
        )

    if escalation_needed:
        system += (
            "⚠ ESCALATION TRIGGER:\n"
            "The patient said your previous responses weren't helpful. This response must be noticeably different: "
            "more specific, more committed, with a concrete intervention. "
            "Name the most likely diagnosis. Give exact medication + dose if appropriate. "
            "Tell them exactly when to escalate to urgent care.\n\n"
        )

    if patient_name:
        system += "Patient's first name: " + patient_name + ". Use naturally and sparingly (max once per response).\n\n"
    if lang_instruction:
        system += lang_instruction + "\n\n"

    # Patient history block — clearly demarcated
    system += "<patient_history>\n"
    if memory_context:
        system += "What this patient has explicitly told you in this conversation:\n" + memory_context + "\n"
    else:
        system += "This patient has NOT stated any conditions, medications, or chronic illnesses yet. Do not assume any exist.\n"
    system += "</patient_history>\n\n"

    # Reference knowledge block — clearly labeled as NOT patient-specific
    system += (
        "<reference_knowledge>\n"
        "The following is generic medical information retrieved to help you reason about this query. "
        "It is NOT about the current patient. Do NOT reference it as something the patient said. "
        "Do NOT mention any specific medications, conditions, or stories from this section unless the patient has independently brought them up.\n\n"
        + clean_context + "\n"
        "</reference_knowledge>\n"
    )

    full_response = ""
    stream_error = None

    # Try Claude first (primary)
    if CLAUDE_ACTIVE:
        try:
            # Anthropic expects system prompt as a separate arg and only user/assistant messages
            anthropic_messages = history + [{"role": "user", "content": question}]
            # Ensure messages alternate user/assistant and start with user
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
            # Claude succeeded, emit done and return
            yield ("done", full_response, {"memory": memory, "sources": sources, "confidence": confidence_level, "confidence_pct": confidence_pct, "engine": "claude"})
            return
        except Exception as e:
            stream_error = e
            print("Claude stream failed, falling back to Groq:", e)
            full_response = ""  # reset so Groq gets a clean start

    # Fallback to Groq (if Claude not configured or failed)
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
    r = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "You are MediChat, a warm clinical AI assistant. Analyse this medical image. Provide: Clinical Observations, Possible Conditions, Recommendations. Use simple compassionate language. Always recommend consulting a doctor." + memory_note + lang_note + "\n\nQuestion: " + prompt},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}
        ]}],
        temperature=0.5, max_tokens=1024
    )
    return r.choices[0].message.content

def clean_text(text):
    replacements = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2022": "-"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")

def generate_chat_pdf(messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    pdf.set_fill_color(15, 118, 110)
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
            pdf.set_fill_color(13, 148, 136)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 7, "  You", ln=True, fill=True)
            pdf.set_fill_color(240, 253, 250)
            pdf.set_text_color(19, 78, 74)
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
    pdf.set_fill_color(15, 118, 110)
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
        pdf.set_fill_color(15, 118, 110)
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

with st.sidebar:
    st.markdown("## MediChat")
    st.markdown("---")

    # Language selector
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
    st.markdown('<div class="sb-footer">MediChat v4.0<br>ICT654 - Group 7 - SISTC 2026</div>', unsafe_allow_html=True)

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

# Trust strip: compact, only 3 key signals
st.markdown(
    '<div class="trust-strip">'
    '<span class="trust-pill"><span class="trust-pill-icon">🔒</span>Private</span>'
    '<span class="trust-pill"><span class="trust-pill-icon">📚</span>1,000 medical sources</span>'
    '<span class="trust-pill"><span class="trust-pill-icon">✓</span>Evidence-based</span>'
    '</div>',
    unsafe_allow_html=True
)

L = LANGUAGES[st.session_state.selected_language]

# Admin access: URL parameter triggers the password gate.
# Patients cannot see the Analytics tab at all unless they:
#   1. Know the admin URL parameter (?admin=1)
#   2. Enter the correct password
ADMIN_PASSWORD = "MediChat@Group7#2026"  # Change this before final submission
_query_params = st.query_params
_admin_requested = _query_params.get("admin", "") != ""

# Show password gate if admin URL is visited but not yet authenticated
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

if _is_admin:
    # Admin logout option at top
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
    # Patient view: only Free Chat and Symptom Check, no Analytics
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

# Emergency banner - shown whenever emergency keywords detected in this session
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

    if not st.session_state.messages:
        if not st.session_state.patient_name:
            # Ask for name first
            st.markdown(
                '<div class="welcome-card">'
                '<div style="font-size:3rem;margin-bottom:0.7rem;">👋</div>'
                '<div class="welcome-title">' + L["greeting"] + '</div>'
                '<div class="welcome-text">Before we start, what should I call you? Sharing your name is optional but helps me personalise our conversation.</div>'
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
                st.rerun()
            if skip_name:
                st.session_state.patient_name = "Guest"
                st.rerun()
        else:
            # Show personalised welcome
            display_name = "" if st.session_state.patient_name == "Guest" else ", " + st.session_state.patient_name
            st.markdown(
                '<div class="welcome-card">'
                '<div style="font-size:3rem;margin-bottom:0.7rem;">👋</div>'
                '<div class="welcome-title">Hi' + display_name + '! How can I help you today?</div>'
                '<div class="welcome-text">' + L["welcome_text"] + '</div>'
                '<div class="chip-row">'
                '<span class="chip">Medications</span>'
                '<span class="chip">Heart Health</span>'
                '<span class="chip">Conditions</span>'
                '<span class="chip">Nutrition</span>'
                '<span class="chip">Mental Health</span>'
                '<span class="chip">Infections</span>'
                '</div></div>',
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
                st.markdown('<div class="bot-wrap"><div class="av av-bot">M</div><div class="bot-bubble">' + content + '</div></div>', unsafe_allow_html=True)
                # Engine badge (Claude Haiku vs Groq fallback)
                engine_used = msg.get("engine", "")
                engine_html = ""
                if engine_used == "claude":
                    engine_html = '<span class="engine-badge engine-claude">Claude Haiku</span>'
                elif engine_used == "groq":
                    engine_html = '<span class="engine-badge engine-groq">Llama (fallback)</span>'
                # Show source tags alongside engine badge
                msg_sources = msg.get("sources", [])
                source_tags = "".join(['<span class="source-tag">📚 ' + s + '</span>' for s in msg_sources])
                if engine_html or source_tags:
                    st.markdown('<div class="source-row">' + engine_html + source_tags + '</div>', unsafe_allow_html=True)
                # Show confidence indicator
                conf_level = msg.get("confidence")
                conf_pct = msg.get("confidence_pct")
                if conf_level and conf_pct:
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
        if st.button(L["download_chat_btn"], use_container_width=True):
            pdf_bytes = generate_chat_pdf(st.session_state.messages)
            st.download_button(label="Click here to save your PDF", data=pdf_bytes, file_name="MediChat_Conversation_" + datetime.now().strftime("%Y%m%d_%H%M") + ".pdf", mime="application/pdf", use_container_width=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">' + L["upload_label"] + '</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload medical image or PDF report",
        type=["jpg", "jpeg", "png", "pdf"],
        label_visibility="collapsed",
        key="uploader_" + str(st.session_state.uploader_key),
        help="Upload a medical image (X-ray, skin condition, etc.) or a PDF report (blood test, lab results, etc.)"
    )
    if uploaded_image:
        is_pdf_upload = uploaded_image.name.lower().endswith(".pdf")
        if is_pdf_upload:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.7rem;padding:0.8rem 1rem;background:var(--sage-50);border:1px solid var(--sage-100);border-radius:12px;margin:0.5rem 0;">'
                '<div style="width:40px;height:40px;background:var(--sage-500);color:white;border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.7rem;">PDF</div>'
                '<div style="flex:1;">'
                '<div style="font-weight:600;font-size:0.88rem;color:var(--sage-900);">' + uploaded_image.name + '</div>'
                '<div style="font-size:0.75rem;color:var(--warm-gray);">Ready for analysis. Ask MediChat what you want to know about this report.</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            ia, ib, ic = st.columns([1, 2, 1])
            with ib:
                st.image(uploaded_image, caption="Ready for analysis", use_column_width=True)
    st.markdown('<div class="section-label" style="margin-top:0.7rem;">' + L["question_label"] + '</div>', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("", placeholder=L["placeholder"], label_visibility="collapsed")
        bc1, bc2, bc3, bc4 = st.columns([1, 2, 2, 1])
        with bc2:
            submit = st.form_submit_button(L["send_btn"], use_container_width=True)
        with bc3:
            clear = st.form_submit_button(L["clear_btn"], use_container_width=True)
    st.markdown('<div class="disclaimer-mini disclaimer-mini-red">⚠ MediChat is not a substitute for professional medical advice. For diagnosis or treatment, please consult a qualified doctor.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
        st.rerun()

    if submit and (user_input.strip() or uploaded_image):
        st.session_state.qcount += 1
        lang_instruction = LANGUAGES[st.session_state.selected_language]["lang_instruction"]

        # Advanced emergency detection: check user input + full conversation history
        if user_input.strip():
            conv_text = " ".join([m.get("content", "") for m in st.session_state.messages if m.get("type") == "text"])
            is_emerg, reason = detect_emergency(user_input, conv_text)
            if is_emerg:
                st.session_state.emergency_detected = True
                st.session_state.emergency_reason = reason

        if uploaded_image:
            is_pdf = uploaded_image.name.lower().endswith(".pdf")
            if is_pdf:
                st.session_state.messages.append({"role": "user", "type": "pdf", "content": user_input.strip() + " [PDF: " + uploaded_image.name + "]"})
                with st.spinner("Reading your medical report..."):
                    pdf_text = extract_pdf_text(uploaded_image)
                    if not pdf_text:
                        reply = "I had trouble reading that PDF. It might be image-based (scanned) rather than text-based. Could you try uploading it as a JPEG or PNG image instead?"
                        engine_used = "system"
                    else:
                        reply, engine_used = medichat_pdf_analysis(user_input, pdf_text, st.session_state.messages, lang_instruction)
                        reply = strip_excessive_disclaimers(reply)
                st.session_state.last_sources = ["PDF Report Analysis"]
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply, "sources": st.session_state.last_sources, "confidence": "medium", "confidence_pct": 75, "engine": engine_used})
            else:
                st.session_state.messages.append({"role": "user", "type": "image", "content": user_input.strip()})
                with st.spinner("Analysing your image..."):
                    uploaded_image.seek(0)
                    reply = medichat_vision(user_input, encode_image(uploaded_image), st.session_state.messages, lang_instruction)
                    reply = strip_excessive_disclaimers(reply)
                st.session_state.last_sources = ["Vision AI (Llama-4-Scout)"]
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply, "sources": st.session_state.last_sources, "confidence": "medium", "confidence_pct": 75, "engine": "groq-vision"})
        else:
            user_msg = {"role": "user", "type": "text", "content": user_input.strip()}
            st.session_state.messages.append(user_msg)
            import time as _time
            _t0 = _time.time()

            # Determine avatar for live rendering
            user_initial_live = "U"
            if st.session_state.patient_name and st.session_state.patient_name != "Guest":
                user_initial_live = st.session_state.patient_name[0].upper()

            # Render the just-submitted user message immediately
            st.markdown('<div class="user-wrap"><div class="user-bubble">' + user_msg["content"] + '</div><div class="av av-user">' + user_initial_live + '</div></div>', unsafe_allow_html=True)

            # Show thinking indicator
            st.markdown('<div class="bot-label">MediChat</div>', unsafe_allow_html=True)
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(
                '<div class="bot-wrap">'
                '<div class="av av-bot">M</div>'
                '<div class="thinking-indicator">'
                '<div class="thinking-dot"></div>'
                '<div class="thinking-dot"></div>'
                '<div class="thinking-dot"></div>'
                '<span class="thinking-label">thinking</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            name_for_rag = "" if st.session_state.patient_name == "Guest" else st.session_state.patient_name

            # Stream the response progressively
            final_text = ""
            stream_metadata = None
            try:
                for event in medichat_rag_stream(user_input, st.session_state.messages, lang_instruction, name_for_rag):
                    kind = event[0]
                    if kind == "chunk":
                        final_text = event[2]
                        # Update the placeholder with streaming content + blinking cursor
                        thinking_placeholder.markdown(
                            '<div class="bot-wrap">'
                            '<div class="av av-bot">M</div>'
                            '<div class="bot-bubble">' + final_text + '<span class="stream-cursor"></span></div>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    elif kind == "done":
                        final_text = event[1]
                        stream_metadata = event[2]
            except Exception as e:
                thinking_placeholder.empty()
                st.error("MediChat had trouble generating a response. Please try again.")
                st.stop()

            # Strip any inline disclaimer spam before final render
            final_text = strip_excessive_disclaimers(final_text)

            # Final render without cursor
            thinking_placeholder.markdown(
                '<div class="bot-wrap">'
                '<div class="av av-bot">M</div>'
                '<div class="bot-bubble">' + final_text + '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            memory = stream_metadata["memory"]
            sources = stream_metadata["sources"]
            conf_level = stream_metadata["confidence"]
            conf_pct = stream_metadata["confidence_pct"]
            engine_used = stream_metadata.get("engine", "unknown")
            st.session_state.patient_memory = memory
            st.session_state.last_sources = sources
            _response_time = round(_time.time() - _t0, 2)
            st.session_state.response_times.append(_response_time)

            # Drug-condition interaction check
            interaction_alerts = check_drug_interactions(final_text, memory)
            if interaction_alerts:
                alert_block = "\n\n---\n\n**⚠️ Drug Safety Check:**\n"
                for a in interaction_alerts:
                    alert_block += "\n- **" + a["drug"] + "** — given your " + ", ".join(a["conditions"]) + ": " + a["warning"]
                final_text = final_text + alert_block

            # Log for evaluation dashboard (session + cross-session)
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
        st.rerun()

elif st.session_state.mode == "eval":
    # ── Evaluation Dashboard (Admin Only) ──────────────────────────
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

    # Data source toggle: Current session vs All patients (Firestore)
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
            st.caption("⚠️ Firestore not connected. Showing current session only.")

    # Load logs based on selected data source
    if data_source == "All Patients (Firestore)" and FIREBASE_ACTIVE:
        raw_logs = fetch_all_queries_from_firestore(limit=500)
        # Convert Firestore docs to same shape as session logs
        logs = [
            {
                "query": " " * d.get("query_word_count", 0),  # Placeholder for word count, no real text
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
        st.info("📡 Live data from Firestore — aggregated from all MediChat patients (anonymised). Total records: " + str(len(logs)))
    else:
        logs = st.session_state.eval_log
        if FIREBASE_ACTIVE:
            st.caption("💻 Current session only. Switch to 'All Patients (Firestore)' to see aggregate data.")
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
        # Top metrics
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

        # Confidence distribution
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

        # Safety metrics
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

        # Language distribution
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

        # Recent queries log (anonymised - no patient text shown)
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

        # Export dashboard data
        ec1, ec2 = st.columns([1, 1])
        with ec1:
            # Build Excel workbook with multiple sheets
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

                # ── Sheet 1: Summary ──
                ws1 = wb.active
                ws1.title = "Summary"
                ws1["A1"] = "MediChat Analytics Summary"
                ws1["A1"].font = _Font(name="Arial", bold=True, size=16, color="1F3864")
                ws1.merge_cells("A1:B1")
                ws1["A2"] = "Generated: " + datetime.now().strftime("%B %d, %Y at %I:%M %p")
                ws1["A2"].font = _Font(name="Arial", italic=True, size=10, color="555555")
                ws1.merge_cells("A2:B2")
                ws1["A3"] = "Privacy Note: All patient query text has been anonymised. Only aggregated metadata is shown."
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

                # ── Sheet 2: Language Distribution ──
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

                # ── Sheet 3: Anonymised Query Log ──
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

                # ── Sheet 4: Methodology Notes ──
                ws4 = wb.create_sheet("Methodology")
                ws4["A1"] = "MediChat Analytics Methodology"
                ws4["A1"].font = _Font(name="Arial", bold=True, size=14, color="1F3864")
                ws4.merge_cells("A1:B1")

                method_notes = [
                    ("Confidence Score", "Calculated from FAISS L2 distance between query embedding and top-3 retrieved documents. Lower distance = higher confidence. < 0.8 = High, 0.8-1.3 = Medium, > 1.3 = Low."),
                    ("Source Classification", "PubMed Research = documents 0-499 in FAISS index (biomedical research papers). Doctor-Patient Data = documents 500-999 (real clinical conversations)."),
                    ("Emergency Detection", "Hybrid system: direct keyword match on 30+ emergency terms, plus 6 symptom-cluster patterns (cardiac, stroke, anaphylaxis, syncope, severe asthma, hyperglycemic crisis)."),
                    ("Drug Safety Warnings", "Hard-coded rules cross-reference 6 drug classes (antihistamines, NSAIDs, decongestants, paracetamol, bismuth, PPIs) against patient-stated conditions from memory extraction."),
                    ("Response Time", "Measured from query submission to final response display, including FAISS retrieval and Groq LLM inference."),
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

                # Save to buffer
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
