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
import os
import base64
import hashlib
import hmac
import html
import logging
import secrets as _py_secrets
from PIL import Image, ImageEnhance, ImageOps
import io
import re
import time
import difflib
from datetime import datetime, timedelta, timezone, date as _date
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
from fpdf import FPDF

# Firebase for cross-session analytics (optional - fails gracefully if missing)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

logger = logging.getLogger("medichat")
if not logger.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_log_handler)
logger.setLevel(os.environ.get("MEDICHAT_LOG_LEVEL", "INFO"))

def _safe_int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _safe_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default

APP_TITLE = "MediChat Ai"
APP_SUBTITLE = "Your Ai Health Assistant"
APP_VERSION_LABEL = APP_TITLE

def _resolve_asset_path(filename):
    """Locate an asset by filename, enforcing an absolute path relative to this script.
    
    This guarantees reliability across local environments and Streamlit Cloud container restarts.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_dir = os.getcwd()
        
    candidates = [
        os.path.join(base_dir, "assets", filename),
        os.path.join(base_dir, filename),
        os.path.join("assets", filename)
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
            
    # Fallback to prevent unhandled runtime errors if file is created dynamically
    return os.path.join(base_dir, "assets", filename)

@st.cache_data(show_spinner=False)
def _load_asset_data_uri_cached(path, mtime):
    """Cache key is (path, mtime) so the cache invalidates automatically when
    the file changes on disk — no stale None after a missing-asset boot."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    if ext == "jpg":
        ext = "jpeg"
    return "data:image/" + ext + ";base64," + b64

def load_asset_data_uri(filename):
    """Return an asset as a base64 data URI, or None if missing/unreadable."""
    path = _resolve_asset_path(filename)
    if not path:
        return None
    try:
        return _load_asset_data_uri_cached(path, os.path.getmtime(path))
    except Exception:
        return None

def get_brand_logo_data_uri():
    """Backwards-compatible accessor for the MediChat brand logo."""
    return load_asset_data_uri("MediChat logo.png")
MEDICAL_REFERENCE_TARGET = max(1000, _safe_int_env("MEDICHAT_REFERENCE_TARGET", 5000))
PRIVACY_POLICY_URL = _safe_secret(
    "PRIVACY_POLICY_URL",
    os.environ.get("PRIVACY_POLICY_URL", "?mode=privacy"),
)

st.set_page_config(
    page_title=APP_VERSION_LABEL,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Firebase Initialization (cross-session analytics) ────────────────
@st.cache_resource
def init_firebase():
    """Initialize Firebase Admin SDK. Returns Firestore client or None if unavailable."""
    if not FIREBASE_AVAILABLE:
        return None
    try:
        firebase_config = _safe_secret("firebase", {})
        if not firebase_config or not firebase_config.get("project_id"):
            return None
        if not firebase_admin._apps:
            cred = credentials.Certificate(dict(firebase_config))
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        logger.warning("Firebase init failed: %s", e)
        return None

firestore_db = init_firebase()
FIREBASE_ACTIVE = firestore_db is not None

# ── Persistent Patient Profiles (email + PIN, Firestore-backed) ──────
PROFILE_SALT = _safe_secret("PROFILE_SALT", os.environ.get("PROFILE_SALT", "medichat-default-change-me"))

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
        logger.warning("Profile fetch failed: %s", e)
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
        logger.warning("Profile create failed: %s", e)
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
        logger.warning("Profile visit-count update failed: %s", e)
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
        logger.warning("Profile state persist failed: %s", e)

# ── Per-Chat History (each conversation is its own Firestore doc) ────
def _trim_messages_for_storage(messages):
    trimmed = []
    for m in (messages or [])[-80:]:
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
    return trimmed

def derive_chat_title(messages):
    for m in messages or []:
        if m.get("role") == "user" and m.get("content"):
            t = (m["content"] or "").strip().replace("\n", " ")
            return (t[:50] + "…") if len(t) > 50 else t
    return "New chat"

def generate_ai_chat_title(messages):
    """Use Claude to summarise a chat into a 3-6 word clinical title.
    One call per chat. Falls back to derive_chat_title on any failure."""
    if not CLAUDE_ACTIVE or not messages:
        return derive_chat_title(messages)
    try:
        lines = []
        for m in messages[:8]:
            role = "Patient" if m.get("role") == "user" else "MediChat"
            content = (m.get("content", "") or "")[:280]
            if content:
                lines.append(role + ": " + content)
        transcript = "\n".join(lines)
        resp = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=24,
            system=(
                "You are a clinical scribe. Read the chat and output a 3-6 word title "
                "summarising the patient's chief concern. Output the title and nothing else: "
                "no quotes, no punctuation at the end, no preamble. Examples: "
                "Migraine workup, Type 2 diabetes review, Persistent dry cough, "
                "Lower back pain after lifting."
            ),
            messages=[{"role": "user", "content": transcript}],
            temperature=0.3,
        )
        title = (resp.content[0].text or "").strip().strip('"\'').rstrip(".")[:60]
        return title or derive_chat_title(messages)
    except Exception as e:
        logger.warning("AI title generation failed: %s", e)
        return derive_chat_title(messages)

def list_conversations(email_hash, limit=30):
    if not FIREBASE_ACTIVE or not email_hash:
        return []
    try:
        docs = (firestore_db.collection("medichat_profiles")
                .document(email_hash)
                .collection("conversations")
                .order_by("last_updated", direction=firestore.Query.DESCENDING)
                .limit(limit).stream())
        out = []
        for d in docs:
            data = d.to_dict() or {}
            out.append({
                "id": d.id,
                "title": data.get("title", "Chat"),
                "message_count": data.get("message_count", 0),
                "last_updated": data.get("last_updated"),
                "first_user_msg": data.get("first_user_msg", ""),
                "last_assistant_msg": data.get("last_assistant_msg", ""),
            })
        return out
    except Exception as e:
        logger.warning("list_conversations failed: %s", e)
        return []

def load_conversation(email_hash, conv_id):
    if not FIREBASE_ACTIVE or not email_hash or not conv_id:
        return None
    try:
        doc = (firestore_db.collection("medichat_profiles")
               .document(email_hash)
               .collection("conversations")
               .document(conv_id).get())
        if not doc.exists:
            return None
        return doc.to_dict()
    except Exception as e:
        logger.warning("load_conversation failed: %s", e)
        return None

def save_conversation(email_hash, conv_id, messages):
    if not FIREBASE_ACTIVE or not email_hash:
        return None
    trimmed = _trim_messages_for_storage(messages)
    msg_count = len(trimmed)
    # Lightweight summary fields for cross-chat context (avoids re-reading the full doc).
    _first_user = next((m.get("content", "") for m in trimmed if m.get("role") == "user"), "")
    _last_asst = next((m.get("content", "") for m in reversed(trimmed) if m.get("role") == "assistant"), "")
    payload = {
        "messages": trimmed,
        "message_count": msg_count,
        "first_user_msg": (_first_user or "")[:240],
        "last_assistant_msg": (_last_asst or "")[:320],
        "last_updated": firestore.SERVER_TIMESTAMP,
    }
    # Title strategy: cheap fallback on first save, AI upgrade once at 4 messages.
    if not conv_id:
        payload["title"] = derive_chat_title(trimmed)
    elif msg_count == 4:
        payload["title"] = generate_ai_chat_title(trimmed)
    try:
        coll = firestore_db.collection("medichat_profiles").document(email_hash).collection("conversations")
        if conv_id:
            coll.document(conv_id).set(payload, merge=True)
            return conv_id
        # New conversation: include created_at
        payload["created_at"] = firestore.SERVER_TIMESTAMP
        ref = coll.add(payload)
        return ref[1].id if isinstance(ref, tuple) else ref.id
    except Exception as e:
        logger.warning("save_conversation failed: %s", e)
        return None

def delete_conversation(email_hash, conv_id):
    if not FIREBASE_ACTIVE or not email_hash or not conv_id:
        return False
    try:
        (firestore_db.collection("medichat_profiles")
         .document(email_hash)
         .collection("conversations")
         .document(conv_id).delete())
        return True
    except Exception as e:
        logger.warning("delete_conversation failed: %s", e)
        return False

# ── Generic per-user data store (Firestore for auth, session for guest) ─
import uuid as _uuid

GUEST_DATA_KEY = "guest_user_data"

def _ensure_guest_store():
    if GUEST_DATA_KEY not in st.session_state:
        st.session_state[GUEST_DATA_KEY] = {
            "medications": [],
            "appointments": [],
            "health_records": [],
            "daily_metrics": {},
        }
    return st.session_state[GUEST_DATA_KEY]

def _today_key():
    return datetime.now().strftime("%Y-%m-%d")

def get_user_doc():
    """Read fresh profile doc for the signed-in user; returns {} if guest/none."""
    if not (st.session_state.get("is_authenticated") and st.session_state.get("user_email_hash") and FIREBASE_ACTIVE):
        return None
    try:
        snap = firestore_db.collection("medichat_profiles").document(st.session_state.user_email_hash).get()
        return snap.to_dict() or {} if snap.exists else {}
    except Exception as e:
        logger.warning("get_user_doc failed: %s", e)
        return {}

def update_user_doc(updates):
    """Patch the signed-in user's profile doc."""
    if not (st.session_state.get("is_authenticated") and st.session_state.get("user_email_hash") and FIREBASE_ACTIVE):
        return False
    try:
        firestore_db.collection("medichat_profiles").document(st.session_state.user_email_hash).set(updates, merge=True)
        return True
    except Exception as e:
        logger.warning("update_user_doc failed: %s", e)
        return False

# ── Medications ──────────────────────────────────────────────────────
def list_medications():
    if st.session_state.get("is_authenticated"):
        return (get_user_doc() or {}).get("medications", []) or []
    return _ensure_guest_store()["medications"]

def add_medication(name, dose, frequency, time_of_day, notes=""):
    name = (name or "").strip()
    if not name:
        return False
    entry = {
        "id": str(_uuid.uuid4())[:12],
        "name": name[:80],
        "dose": (dose or "").strip()[:40],
        "frequency": (frequency or "Once daily")[:30],
        "time_of_day": (time_of_day or "")[:30],
        "notes": (notes or "").strip()[:240],
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    if st.session_state.get("is_authenticated"):
        current = list_medications()
        update_user_doc({"medications": current + [entry]})
    else:
        _ensure_guest_store()["medications"].append(entry)
    return True

def delete_medication(med_id):
    if st.session_state.get("is_authenticated"):
        current = [m for m in list_medications() if m.get("id") != med_id]
        update_user_doc({"medications": current})
    else:
        store = _ensure_guest_store()
        store["medications"] = [m for m in store["medications"] if m.get("id") != med_id]
    return True

# ── Appointments ─────────────────────────────────────────────────────
def list_appointments():
    if st.session_state.get("is_authenticated"):
        return (get_user_doc() or {}).get("appointments", []) or []
    return _ensure_guest_store()["appointments"]

def add_appointment(title, date_iso, doctor, location, notes=""):
    title = (title or "").strip()
    if not title or not date_iso:
        return False
    entry = {
        "id": str(_uuid.uuid4())[:12],
        "title": title[:80],
        "date": date_iso,
        "doctor": (doctor or "").strip()[:60],
        "location": (location or "").strip()[:80],
        "notes": (notes or "").strip()[:240],
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    if st.session_state.get("is_authenticated"):
        current = list_appointments()
        update_user_doc({"appointments": current + [entry]})
    else:
        _ensure_guest_store()["appointments"].append(entry)
    return True

def delete_appointment(appt_id):
    if st.session_state.get("is_authenticated"):
        current = [a for a in list_appointments() if a.get("id") != appt_id]
        update_user_doc({"appointments": current})
    else:
        store = _ensure_guest_store()
        store["appointments"] = [a for a in store["appointments"] if a.get("id") != appt_id]
    return True

# ── Health Records (file metadata only — file content not stored to keep doc <1MB) ──
def list_health_records():
    if st.session_state.get("is_authenticated"):
        return (get_user_doc() or {}).get("health_records", []) or []
    return _ensure_guest_store()["health_records"]

def add_health_record(name, file_type, size_bytes, summary=""):
    name = (name or "").strip()
    if not name:
        return False
    entry = {
        "id": str(_uuid.uuid4())[:12],
        "name": name[:120],
        "file_type": (file_type or "")[:24],
        "size_bytes": int(size_bytes or 0),
        "summary": (summary or "").strip()[:1500],
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    if st.session_state.get("is_authenticated"):
        current = list_health_records()
        update_user_doc({"health_records": current + [entry]})
    else:
        _ensure_guest_store()["health_records"].append(entry)
    return True

def delete_health_record(rec_id):
    if st.session_state.get("is_authenticated"):
        current = [r for r in list_health_records() if r.get("id") != rec_id]
        update_user_doc({"health_records": current})
    else:
        store = _ensure_guest_store()
        store["health_records"] = [r for r in store["health_records"] if r.get("id") != rec_id]
    return True

# ── Daily Metrics (water, sleep) ─────────────────────────────────────
DAILY_METRIC_DEFAULTS = {
    "water_glasses": 0,
    "sleep_hours": None,
    "mood": None,
    "steps": None,
    "heart_rate_resting": None,
}

def get_daily_metrics(date_key=None):
    date_key = date_key or _today_key()
    if st.session_state.get("is_authenticated"):
        all_dm = (get_user_doc() or {}).get("daily_metrics", {}) or {}
    else:
        all_dm = _ensure_guest_store()["daily_metrics"]
    # Merge with defaults so missing keys are explicit None rather than KeyError-prone.
    raw = all_dm.get(date_key, {})
    return {**DAILY_METRIC_DEFAULTS, **raw}

def heart_rate_status(bpm):
    """Return (label, css_status_class) for a resting heart rate, or (None, None)."""
    if bpm is None:
        return (None, None)
    try:
        v = int(bpm)
    except Exception:
        return (None, None)
    if 60 <= v <= 100:
        return ("Normal", "md-status-good")
    if v < 60:
        return ("Low", "md-status-info")
    return ("High", "md-status-warn")

def sleep_status(hours):
    if hours is None:
        return (None, None)
    try:
        v = float(hours)
    except Exception:
        return (None, None)
    if 7 <= v <= 9:
        return ("Good", "md-status-good")
    if 6 <= v < 7 or 9 < v <= 10:
        return ("Fair", "md-status-info")
    return ("Low" if v < 6 else "High", "md-status-warn")

def update_daily_metric(field, value, date_key=None):
    date_key = date_key or _today_key()
    if st.session_state.get("is_authenticated"):
        doc = get_user_doc() or {}
        all_dm = doc.get("daily_metrics", {}) or {}
        day = all_dm.get(date_key, {})
        day[field] = value
        all_dm[date_key] = day
        update_user_doc({"daily_metrics": all_dm})
    else:
        store = _ensure_guest_store()
        day = store["daily_metrics"].get(date_key, {})
        day[field] = value
        store["daily_metrics"][date_key] = day

def get_metrics_history(days=7):
    """Return list of (date_str, metrics_dict) for last N days."""
    if st.session_state.get("is_authenticated"):
        all_dm = (get_user_doc() or {}).get("daily_metrics", {}) or {}
    else:
        all_dm = _ensure_guest_store()["daily_metrics"]
    out = []
    for i in range(days - 1, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        out.append((d, {**DAILY_METRIC_DEFAULTS, **all_dm.get(d, {})}))
    return out

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
        logger.warning("Firestore write failed: %s", e)

def fetch_all_queries_from_firestore(limit=500):
    """Retrieve all anonymised query logs for admin dashboard."""
    if not FIREBASE_ACTIVE:
        return []
    try:
        docs = firestore_db.collection("medichat_queries").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        logger.warning("Firestore read failed: %s", e)
        return []

def ui_escape(value):
    """Escape dynamic text before placing it inside custom HTML."""
    return html.escape(str(value or ""), quote=True)

def ui_text(value, max_chars=None):
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if max_chars and len(text) > max_chars:
        text = text[:max_chars - 3].rstrip() + "..."
    return ui_escape(text)

def ui_lines(value):
    return ui_escape(value).replace("\n", "<br>")

def get_user_local_now():
    tz_name = ""
    try:
        tz_name = getattr(st.context, "timezone", "") or ""
    except Exception:
        tz_name = ""
    if tz_name and ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(tz_name))
        except Exception:
            pass
    return datetime.now()

def _msg_now_ts():
    """Return the current user-local time as a short '10:21 AM' string,
    used as a per-message timestamp for new conversations going forward.
    Old messages stored without a ts will simply skip the timestamp line."""
    try:
        return get_user_local_now().strftime("%-I:%M %p")
    except Exception:
        try:
            return get_user_local_now().strftime("%I:%M %p").lstrip("0")
        except Exception:
            return ""

def reset_prescription_reader_state():
    st.session_state.rx_reader_result = None
    st.session_state.rx_uploader_key = st.session_state.get("rx_uploader_key", 0) + 1

_NEW_CHAT_RESET = {
    "current_conversation_id": "",
    "messages": [],
    "qcount": 0,
    "feedback": {},
    "last_sources": [],
    "last_pdf_context": "",
    "last_image_context": "",
    "emergency_detected": False,
    "triage_assessment": None,
    "pending_user_input": "",
    "home_show_vision_upload": False,
    "home_show_voice": False,
    "assessment_stage": 0,
    "assessment_data": {},
    "assessment_complete": False,
    "assessment_report": None,
    "assessment_parsed": None,
    "mode": "chat",
}
_NEW_CHAT_BUMP = ("chat_input_key", "uploader_key", "voice_audio_key")

def start_new_chat_session():
    for k, v in _NEW_CHAT_RESET.items():
        st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v
    for k in _NEW_CHAT_BUMP:
        st.session_state[k] = st.session_state.get(k, 0) + 1
    reset_prescription_reader_state()

# ── Stylesheet loader (was 11 inline st.markdown CSS blocks) ──────
def _load_app_css():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_dir = os.getcwd()
    css_path = os.path.join(base_dir, "assets", "style.css")
    try:
        with open(css_path, "r") as _f:
            css = _f.read()
        st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)
    except Exception as _css_err:
        logger.warning("Failed to load assets/style.css: %s", _css_err)

_load_app_css()

# ── Dashboard Reskin (overrides above via cascade) ────────────────────

# ── Final premium cleanup pass (reference-aligned) ─────────────────────

# ── UX refinement pass (final cascade layer) ─────────────────────────

# ── Client polish pass: sidebar, guest, cards, home controls ───────────

# ── Annotated QA polish pass (wins over all prior Streamlit/CSS output) ──

# ── Sidebar lock: Streamlit's native collapse button can hide navigation ──

# ── Mockup parity overrides (home dashboard visual match) ─────────────

# ── Final Sidebar Seam Polish ─────────────────────────────────────────

# ── Sidebar Hard-Lock Layout (strict final pass) ─────────────────────

# ── Cross-Profile Main UI Lock (guest + signed-in, not page-specific) ─────────

GROQ_API_KEY = _safe_secret("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
GROQ_ACTIVE = bool(GROQ_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_ACTIVE else None

ANTHROPIC_API_KEY = _safe_secret("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
CLAUDE_MODEL = _safe_secret("CLAUDE_MODEL", "claude-haiku-4-5")
anthropic_client = None
if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
    try:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        logger.warning("Anthropic init failed: %s", e)
        anthropic_client = None

CLAUDE_ACTIVE = anthropic_client is not None

def transcribe_voice_note(audio_file):
    """Transcribe a browser-recorded or uploaded audio note with Groq Whisper."""
    if not GROQ_ACTIVE or groq_client is None:
        return ""
    if not audio_file:
        return ""
    try:
        audio_file.seek(0)
    except Exception:
        pass
    try:
        audio_bytes = audio_file.read()
    except Exception:
        try:
            audio_bytes = audio_file.getvalue()
        except Exception:
            audio_bytes = b""
    if not audio_bytes:
        return ""
    file_name = getattr(audio_file, "name", "voice-note.wav") or "voice-note.wav"
    mime_type = getattr(audio_file, "type", "audio/wav") or "audio/wav"
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(file_name, audio_bytes, mime_type),
            model=os.environ.get("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo"),
            response_format="json",
            temperature=0,
        )
        return (getattr(transcription, "text", "") or "").strip()
    except Exception as e:
        logger.warning("Voice transcription failed: %s", e)
        return ""

# ── Language Config ───────────────────────────────────────────────────
LANGUAGES = {
    "English": {
        "flag": "🇦🇺",
        "greeting": "Hello! How can I help you today?",
        "welcome_text": "I am MediChat, your friendly Ai health assistant.<br>I remember everything you tell me during our conversation.<br><br>Or switch to Symptom Check for a guided assessment!",
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
        "welcome_text": "நான் MediChat, உங்கள் நட்பான Ai சுகாதார உதவியாளர்.<br>உரையாடலில் நீங்கள் சொல்வதை நான் நினைவில் வைத்திருப்பேன்.<br><br>வழிகாட்டப்பட்ட மதிப்பீட்டிற்கு அறிகுறி சரிபார்ப்புக்கு மாறலாம்!",
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
        "welcome_text": "මම MediChat, ඔබේ මිත්‍රශීලී Ai සෞඛ්‍ය සහායකයා.<br>ඔබ කියන සෑම දෙයක්ම මම මතක තබා ගනිමි.<br><br>මඟ පෙන්වූ තක්සේරු කිරීම සඳහා රෝග ලක්ෂණ පරීක්ෂාවට මාරු වන්න!",
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
        "welcome_text": "मैं MediChat हूं, आपका मित्रवत Ai स्वास्थ्य सहायक.<br>आप जो कुछ भी बताते हैं मैं याद रखता हूं.<br><br>निर्देशित मूल्यांकन के लिए लक्षण जांच पर स्विच करें!",
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
        "welcome_text": "ഞാൻ MediChat, നിങ്ങളുടെ സൗഹൃദ Ai ആരോഗ്യ സഹായി.<br>നിങ്ങൾ പറയുന്നതെല്ലാം ഞാൻ ഓർത്തിരിക്കും.<br><br>ഗൈഡഡ് അസസ്മെൻ്റിനായി സിംപ്റ്റം ചെക്കിലേക്ക് മാറുക!",
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

@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Initializes embeddings and mounts the clinical database flat index with strict offline fallbacks."""
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning("Embedding transformer initialization failed, using local model mirror: %s", e)
        return None, None, []

    pubmed_docs = []
    dialog_docs = []
    pubmed_target = max(500, MEDICAL_REFERENCE_TARGET // 2)
    dialog_target = max(500, MEDICAL_REFERENCE_TARGET - pubmed_target)
    
    # Attempt PubMedQA pipeline loading
    try:
        pubmed = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:" + str(pubmed_target) + "]", timeout=10)
        pubmed_docs = ["[PubMed Research]\nQuestion: " + i["question"] + "\nAnswer: " + i["long_answer"] for i in pubmed]
    except Exception as e:
        logger.warning("PubMedQA repository offline. Switched to secure backup arrays: %s", e)

    # Attempt MedDialog pipeline loading
    dialog_sources = [
        ("BinKhoaLe1812/MedDialog-EN-100k", "train[:" + str(dialog_target) + "]", "input", "output"),
        ("shibing624/medical", "train[:" + str(dialog_target) + "]", "instruction", "output")
    ]
    
    for ds_name, ds_split, in_key, out_key in dialog_sources:
        try:
            meddialog = load_dataset(ds_name, split=ds_split, timeout=10)
            dialog_docs = [
                "[Doctor-Patient Conversation]\nPatient: " + str(i.get(in_key, "")) + "\nDoctor: " + str(i.get(out_key, "")) 
                for i in meddialog if i.get(in_key) and i.get(out_key)
            ]
            if dialog_docs:
                break
        except Exception:
            continue

    docs = pubmed_docs + dialog_docs
    
    # Production Safeguard: If all external datasets fail, mount the complete core clinical fallback matrix
    if not docs:
        docs = [
            "[PubMed Research]\nQuestion: What causes severe headaches?\nAnswer: Headaches stem from muscle tension, severe dehydration, stress, vascular migraines, or blood pressure issues. Sudden, thunderclap headaches require urgent screening to rule out subarachnoid haemorrhage or clinical meningitis.",
            "[PubMed Research]\nQuestion: What are the cardinal markers of acute myocardial infarction?\nAnswer: Clinical indicators include crushing central chest tightness, radiating jaw or left arm discomfort, acute dyspnea, diaphoresis, and sudden nausea. This layout requires immediate 000 activation.",
            "[PubMed Research]\nQuestion: How is chronic primary hypertension audited?\nAnswer: Audiological management incorporates radical dietary reductions, structural aerobic exercise, and pharmacological management via ACE inhibitors, calcium channel blockers, or beta-blockers."
        ]

    try:
        embeddings = embedder.encode(docs, show_progress_bar=False)
        idx = faiss.IndexFlatL2(embeddings.shape[1])
        idx.add(embeddings.astype("float32"))
        return embedder, idx, docs
    except Exception as initialization_error:
        logger.warning("FAISS structural assembly failure: %s", initialization_error)
        return None, None, docs

with st.spinner("Loading MediChat knowledge base..."):
    try:
        embedder, index, documents = load_rag_system()
    except Exception as e:
        st.error("MediChat knowledge base failed to load. Please refresh the page in a moment. Error: " + str(e))
        st.stop()

MEDICAL_REFERENCE_COUNT = len(documents)

AU_TOP_DRUGS_FALLBACK = {
    "amoxicillin", "amoxicillin clavulanate", "azithromycin", "cephalexin", "doxycycline", "trimethoprim",
    "metformin", "gliclazide", "empagliflozin", "sitagliptin", "insulin glargine",
    "atorvastatin", "rosuvastatin", "simvastatin", "ezetimibe",
    "amlodipine", "perindopril", "lisinopril", "ramipril", "losartan", "valsartan", "metoprolol",
    "aspirin", "clopidogrel", "apixaban", "rivaroxaban", "warfarin",
    "salbutamol", "budesonide", "fluticasone", "tiotropium", "montelukast",
    "levothyroxine", "prednisone", "prednisolone", "omeprazole", "esomeprazole", "pantoprazole",
    "ibuprofen", "naproxen", "diclofenac", "paracetamol", "codeine", "tramadol",
    "sertraline", "escitalopram", "fluoxetine", "mirtazapine", "venlafaxine",
    "quetiapine", "olanzapine", "risperidone", "lithium",
    "ondansetron", "metoclopramide", "loperamide", "dimenhydrinate",
    "nitrofurantoin", "ciprofloxacin", "valaciclovir", "acyclovir",
}

def load_known_drugs():
    csv_path = _safe_secret("AU_DRUGS_CSV_PATH", os.environ.get("AU_DRUGS_CSV_PATH", ""))
    known = set(AU_TOP_DRUGS_FALLBACK)
    if not csv_path:
        return known
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                token = (line or "").strip().lower()
                if token and token not in {"drug", "name", "medication"}:
                    known.add(token)
    except Exception as e:
        logger.warning("Known-drug list load failed: %s", e)
    return known

KNOWN_DRUGS = load_known_drugs()

PRESCRIPTION_PROMPT = """You are a clinical transcription assistant. Your only job is to read the prescription image and transcribe what is written.
You are NOT giving medical advice, NOT diagnosing, and NOT prescribing treatment.

Read this order:
1. Prescriber/clinic header
2. Patient identifiers
3. Medication line(s)
4. Dose, frequency, route, quantity, repeats
5. Date and signature

Output this exact format:

**MEDICATION**
Reading: <best transcription>
Matches known drug: <yes/no>
Confidence: high | medium | low

**STRENGTH / DOSE**
Reading: <exact text or not specified>
Confidence: high | medium | low

**FREQUENCY / DIRECTIONS**
As written: <exact text>
Plain English: <short rewrite>
Confidence: high | medium | low

**ROUTE**
Reading: <oral/topical/inhaled/injection/not specified>

**QUANTITY**
Reading: <exact text or not specified>

**REFILLS**
Reading: <exact text or not specified>

**PRESCRIBER**
Name: <as written or unclear>
Date: <as written or unclear>

**OVERALL CONFIDENCE**: high | medium | low
**ILLEGIBLE SECTIONS**: <list regions that are genuinely unreadable>

**IMPORTANT**: Transcription only. Do not use this output as dosing or diagnosis advice. Confirm with a pharmacist or prescriber.
"""

def preprocess_prescription(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    max_dim = 1568
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92)
    return out.getvalue()

def validate_drug_name(reading):
    candidate = (reading or "").lower().strip()
    if not candidate:
        return {"match": "none"}
    if candidate in KNOWN_DRUGS:
        return {"match": "exact", "drug": candidate}
    close = difflib.get_close_matches(candidate, KNOWN_DRUGS, n=3, cutoff=0.76)
    if close:
        return {"match": "close", "candidates": close}
    return {"match": "none"}

def extract_medication_reading(transcribed_text):
    m = re.search(r"\*\*MEDICATION\*\*.*?Reading:\s*(.+)", transcribed_text or "", flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    line = (m.group(1) or "").splitlines()[0]
    return line.strip()[:120]

def read_prescription(image_bytes, user_note="", lang_instruction=""):
    processed = preprocess_prescription(image_bytes)
    b64 = base64.standard_b64encode(processed).decode("utf-8")
    prompt = PRESCRIPTION_PROMPT + ("\n\n" + lang_instruction if lang_instruction else "")
    if user_note and user_note.strip():
        prompt += "\n\nContext note from user: " + user_note.strip()[:200]

    reading = ""
    model_used = "unavailable"
    if CLAUDE_ACTIVE:
        try:
            first = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1600,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=0.2,
            )
            reading = first.content[0].text
            model_used = CLAUDE_MODEL
        except Exception as e:
            logger.warning("Prescription reader Claude pass failed: %s", e)

    if not reading:
        try:
            r = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                    ],
                }],
                temperature=0.2,
                max_tokens=1800,
            )
            reading = r.choices[0].message.content
            model_used = "groq-vision"
        except Exception as e:
            logger.warning("Prescription reader Groq pass failed: %s", e)
            return {"reading": "I could not process this prescription image right now. Please try again with a clearer photo.", "model_used": "error", "overall_confidence": "low"}

    low_conf = "overall confidence**: low" in reading.lower() or "overall confidence: low" in reading.lower()
    if low_conf and CLAUDE_ACTIVE:
        try:
            second = anthropic_client.messages.create(
                model=os.environ.get("CLAUDE_RX_ESCALATION_MODEL", "claude-opus-4-7"),
                max_tokens=2200,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": prompt + "\n\nA prior pass was low confidence. Re-check carefully using common Australian prescriptions."},
                    ],
                }],
                temperature=0.2,
            )
            reading = second.content[0].text
            model_used = os.environ.get("CLAUDE_RX_ESCALATION_MODEL", "claude-opus-4-7")
        except Exception as e:
            logger.warning("Prescription reader escalation failed: %s", e)

    med_reading = extract_medication_reading(reading)
    validation = validate_drug_name(med_reading)
    if validation.get("match") == "close":
        reading += "\n\n**Drug name check:** Closest known AU medications: " + ", ".join(validation.get("candidates", []))
    elif validation.get("match") == "exact":
        reading += "\n\n**Drug name check:** Exact match to known AU medication: " + validation.get("drug", "")

    overall = "medium"
    overall_match = re.search(r"\*\*OVERALL CONFIDENCE\*\*\s*:\s*(high|medium|low)", reading, flags=re.IGNORECASE)
    if overall_match:
        overall = overall_match.group(1).lower()

    return {"reading": reading, "model_used": model_used, "overall_confidence": overall, "drug_validation": validation}

def looks_like_prescription_request(text):
    t = (text or "").lower()
    if not t:
        return False
    keys = [
        "prescription", "script", "handwriting", "doctor wrote", "doctor writing",
        "medicine name", "medication name", "can you read this", "what does this say",
        "dose on script", "rx", "chemist",
    ]
    return any(k in t for k in keys)

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
        logger.warning("PDF extraction failed: %s", e)
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
        "- Do not provide a final diagnosis. Use probability language and recommend clinician confirmation.\n"
        "- Do not prescribe medicines, initiate treatment plans, or provide exact dosing changes.\n"
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
            logger.warning("Claude PDF analysis failed, falling back to Groq: %s", e)

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question or "Please review this report and tell me what stands out."}]
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        temperature=0.4,
        max_tokens=1500,
    )
    return r.choices[0].message.content, "groq"

_MEMORY_PHRASES = {
    "symptoms": ["i have","i feel","i am feeling","i've been feeling","i'm experiencing","i suffer from","my back hurts","my chest","my stomach","i feel pain","feeling dizzy","feeling nauseous","i have a fever","i have a cough","i have a headache","shortness of breath","i've been tired","i feel tired","i have pain","i feel sick"],
    "conditions": ["i have diabetes","i have hypertension","i have asthma","i have cancer","i am diabetic","i am hypertensive","i was diagnosed with","i have high blood pressure","i have low blood pressure","i have depression","i have anxiety","i have heart","i have kidney"],
    "medications": ["i am taking","i take","i was prescribed","i'm on","taking medication","prescribed me","i have an inhaler","i take tablets","i take pills"],
}

def _match_first_phrase(content, phrases, bucket):
    for phrase in phrases:
        pos = content.find(phrase)
        if pos == -1:
            continue
        snippet = content[pos:pos+60].strip()
        if snippet and snippet not in bucket:
            bucket.append(snippet)
        return

def extract_patient_memory(messages):
    memory = {key: [] for key in _MEMORY_PHRASES}
    for msg in messages:
        if msg.get("role") != "user" or msg.get("type") != "text":
            continue
        content = (msg.get("content") or "").lower()
        for key, phrases in _MEMORY_PHRASES.items():
            _match_first_phrase(content, phrases, memory[key])
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

# Patterns that indicate the message is meta/docs/test rather than a real
# symptom report. Used to suppress emergency + triage detection on pasted
# instructions, code blocks, or app feedback.
META_INDICATORS = [
    "streamlit", "redeploy", "redeploying", "deployment", "deploy",
    "tier 1", "tier 2", "tier 3", "tier 4", "tier 5",
    "test path", "test plan", "test scenario", "test case",
    "feature", "merging", "merged", "commit", "git push", "branch",
    "github", "claude.ai/code", "session_01", "session_state",
    "pull request", " pr #", "rebase",
    "what you'll see", "what you’ll see",
    "shipped to main", "redeploy in",
]

def is_meta_text(text):
    if not text:
        return False
    t = text.lower()
    score = sum(1 for ind in META_INDICATORS if ind in t)
    has_md = ("**" in t) or ("###" in t) or ("```" in t)
    if has_md:
        score += 1
    if len(t) > 1500:
        score += 1
    return score >= 2

def detect_emergency(text, conversation_text=""):
    if is_meta_text(text):
        return False, None
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

# ── Triage tier scoring (Manchester-style 5-tier) ────────────────────
URGENT_KEYWORDS = [
    "severe pain", "worst pain", "intense pain", "unbearable pain",
    "high fever", "fever over 39", "fever 40", "very high temperature",
    "vomiting blood", "blood in stool", "blood in urine", "coughing up blood",
    "cannot keep anything down", "can't keep food down",
    "severe dehydration", "haven't urinated in", "no urine for",
    "severe headache that won't go", "thunderclap headache",
    "vision changes", "double vision", "lost vision",
    "new confusion", "disoriented", "very confused",
    "severe abdominal pain", "rigid abdomen",
    "rapid heartbeat", "racing heart for hours",
    "severe dizziness", "cannot stand",
    "broken bone", "deformed limb", "bone visible",
    "deep cut", "deep wound", "wound won't stop bleeding",
    "burn larger than", "blistering burn",
    "severe asthma attack", "wheezing badly",
    "severe panic", "cannot stop shaking",
    "pregnancy bleeding", "severe pregnancy pain",
]
CONCERN_KEYWORDS = [
    "fever for", "persistent fever", "fever won't go", "fever lasting",
    "headache for days", "persistent headache", "daily headaches",
    "cough for weeks", "persistent cough", "cough lasting",
    "rash spreading", "rash worse", "growing rash",
    "weight loss", "losing weight", "lost weight",
    "tired all the time", "exhausted constantly", "no energy for weeks",
    "trouble sleeping for", "insomnia for",
    "persistent pain", "pain for days", "pain getting worse",
    "frequent urination", "burning urination", "painful urination",
    "diarrhoea for", "diarrhea for", "constipation for",
    "swelling that won't", "lump that's growing", "new lump",
    "bleeding gums often", "easy bruising",
    "new mole", "mole changing", "changing mole",
    "irregular periods", "missed period",
    "anxious all the time", "feeling depressed", "low mood for weeks",
    "joint pain for", "stiff joints",
    "shortness of breath on stairs", "out of breath easily",
]

def assess_triage_tier(text, conversation_text="", memory=None):
    """Return a 5-tier triage assessment with reasons.
    Tier 1: emergency (call 000). Tier 5: self-care.
    Returns Tier 5 (no banner) for clearly meta / non-symptom text.
    """
    text_l = (text or "").lower()
    conv_l = (conversation_text or "").lower()
    combined = (text_l + " " + conv_l).strip()
    memory = memory or {}

    # Meta detection: skip triage when the user's text is clearly app/docs/test
    # rather than a first-person symptom report. Avoids false positives on
    # pasted release notes, code blocks, or meta questions about MediChat itself.
    if is_meta_text(text):
        return {
            "tier": 5, "label": "Self-care appropriate",
            "icon": "⚪", "color": "#64748b",
            "bg": "linear-gradient(135deg,#64748b,#475569)",
            "next_step": "",
            "reasons": [],
        }

    is_emerg, emerg_reason = detect_emergency(text, conversation_text)
    if is_emerg:
        return {
            "tier": 1,
            "label": "Emergency",
            "icon": "🔴",
            "color": "#dc2626",
            "bg": "linear-gradient(135deg,#dc2626,#991b1b)",
            "next_step": "Call 000 (Australia) or your local emergency number now. Do not wait. Do not drive yourself.",
            "reasons": [emerg_reason or "Emergency pattern detected"],
        }

    matched_urgent = [kw for kw in URGENT_KEYWORDS if kw in combined]
    if matched_urgent:
        return {
            "tier": 2,
            "label": "Urgent care today",
            "icon": "🟠",
            "color": "#ea580c",
            "bg": "linear-gradient(135deg,#ea580c,#c2410c)",
            "next_step": "See a GP, urgent care clinic, or ED today, ideally within 2-4 hours.",
            "reasons": matched_urgent[:3],
        }

    matched_concern = [kw for kw in CONCERN_KEYWORDS if kw in combined]
    has_chronic = bool(memory.get("conditions"))
    if matched_concern or (has_chronic and any(s in combined for s in ["worse", "new symptom", "different"])):
        reasons = matched_concern[:3] if matched_concern else ["Chronic condition + new or worsening symptom"]
        return {
            "tier": 3,
            "label": "GP within 24-48 hrs",
            "icon": "🟡",
            "color": "#ca8a04",
            "bg": "linear-gradient(135deg,#ca8a04,#a16207)",
            "next_step": "Book a GP appointment for today or tomorrow. Watch for worsening signs.",
            "reasons": reasons,
        }

    has_symptoms = bool(memory.get("symptoms")) or any(w in combined for w in ["pain", "ache", "sore", "tired", "dizzy", "nausea"])
    if has_symptoms:
        return {
            "tier": 4,
            "label": "GP within a week",
            "icon": "🟢",
            "color": "#16a34a",
            "bg": "linear-gradient(135deg,#16a34a,#15803d)",
            "next_step": "Book a routine GP appointment in the next few days to discuss your symptoms.",
            "reasons": (memory.get("symptoms") or [])[:3] or ["Active symptoms reported"],
        }

    return {
        "tier": 5,
        "label": "Self-care appropriate",
        "icon": "⚪",
        "color": "#64748b",
        "bg": "linear-gradient(135deg,#64748b,#475569)",
        "next_step": "Self-care and monitoring is reasonable. Reach out if anything new or worsening.",
        "reasons": [],
    }

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
    pubmed_split = max(1, len(documents) // 2)
    pubmed_count = sum(1 for i in idxs if i < pubmed_split)
    dialog_count = sum(1 for i in idxs if i >= pubmed_split)
    sources = []
    if pubmed_count > 0:
        sources.append("PubMed Research (" + str(pubmed_count) + ")")
    if dialog_count > 0:
        sources.append("Doctor-Patient Data (" + str(dialog_count) + ")")
    return sources

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
        html_text = _md.markdown(html.escape(raw, quote=False))
    except ImportError:
        safe = html.escape(raw)
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

def is_casual_message(text):
    if not text:
        return True
    val = text.strip().lower().rstrip("?.! ")
    greetings = {
        "hi", "hello", "hey", "hola", "g'day", "good morning", "good afternoon", "good evening", 
        "howdy", "hi there", "hello there", "yo", "sup", "what's up", "whats up", "what's happening", "whats happening"
    }
    if val in greetings:
        return True
    casual_queries = {
        "how are you", "how are you doing", "how's it going", "hows it going", "what's new", "whats new",
        "who are you", "what is your name", "what are you", "are you there", "tell me a joke", 
        "thanks", "thank you", "ok", "okay"
    }
    if val in casual_queries:
        return True
    return False

def medichat_rag_stream(question, all_messages, lang_instruction="", patient_name="", pdf_context="", image_context="", past_chats_summary=""):
    if is_casual_message(question):
        clean_context = ""
        sources = []
        confidence_level = None
        confidence_pct = 0
    else:
        emb = embedder.encode([question]).astype("float32")
        distances, idxs = index.search(emb, k=6)
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

        "RULE 10 — CLINICAL GUARDRAILS:\n"
        "Do not give a final diagnosis. Do not prescribe medicine. Do not provide exact dosage calculations or dose changes. "
        "Use risk-aware language, safety-net advice, and escalation triggers.\n\n"
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
        system += "What this patient has told you across all your conversations with them:\n" + memory_context + "\n"
    else:
        system += "This patient has NOT stated any conditions, medications, or chronic illnesses yet. Do not assume any exist.\n"
    system += "</patient_history>\n\n"

    if past_chats_summary:
        system += (
            "<past_conversations>\n"
            "This patient has had earlier separate conversations with you. Use these to recognise references "
            "to 'last time' or 'remember when'. Each entry shows when the chat happened, its main topic, and a short summary.\n"
            + past_chats_summary +
            "\n</past_conversations>\n\n"
            "When the patient asks if you remember a past chat, refer to <past_conversations> by topic naturally. "
            "Do not say 'I have no memory'. You DO have access to the topics and summaries listed above.\n\n"
        )

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
            logger.warning("Claude stream failed, falling back to Groq: %s", e)
            full_response = ""

    if groq_client is None:
        raise RuntimeError(
            "No AI backend is available. Check that GROQ_API_KEY (and optionally ANTHROPIC_API_KEY) "
            "are set in your Streamlit Cloud secrets."
        )

    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}]
    try:
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            temperature=0.55,
            max_tokens=1024,
            stream=True,
            timeout=30.0,
        )
    except Exception as stream_exception:
        import traceback as _tb
        logger.warning("Groq streaming error: %s", stream_exception)
        _tb.print_exc()
        raise RuntimeError(f"Groq API error: {str(stream_exception)}") from stream_exception
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
        "- Do not provide a final diagnosis or medication dosage instructions.\n"
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
            logger.warning("Claude vision failed, falling back to Groq: %s", e)

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
            logger.warning("Claude summary failed, falling back to Groq: %s", e)

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
            logger.warning("Groq summary failed: %s", e)
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
    pdf.multi_cell(0, 4, "MediChat is a research prototype, not a medical device. This summary is generated by Ai from a chat conversation and may contain errors or omissions. Always rely on your qualified healthcare professional for diagnosis and treatment decisions.")
    pdf.ln(2)
    pdf.cell(0, 4, APP_VERSION_LABEL, ln=True, align="C")

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
    pdf.cell(0, 5, APP_VERSION_LABEL, align="C")
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
    info_row("Known Conditions", data.get("known_conditions", "Not specified"))
    info_row("Current Medications", data.get("current_medications", "Not specified"))
    info_row("Red-flag Symptoms", data.get("red_flags", "Not specified"))
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
    pdf.multi_cell(0, 6, "IMPORTANT DISCLAIMER: This report was generated by MediChat Ai and does NOT constitute a medical diagnosis. Please share this with your doctor for professional evaluation and treatment.", fill=True)
    pdf.set_y(-18)
    pdf.set_text_color(148, 163, 184)
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(0, 5, APP_VERSION_LABEL, align="C")
    return bytes(pdf.output())

ASSESSMENT_STAGES = [
    {"key": "main_symptom", "question": "What is your main symptom or health concern today?", "hint": "Please describe clearly e.g. chest pain, headache, fever, shortness of breath, dizziness...", "options": ["Chest pain", "Headache", "Fever", "Cough", "Dizziness", "Stomach pain", "Fatigue", "Other (type below)"]},
    {"key": "duration", "question": "How long have you been experiencing this?", "hint": "", "options": ["Just started today", "A few days", "About a week", "More than 2 weeks", "Over a month"]},
    {"key": "severity", "question": "How severe is it on a scale of 1 to 10? (1 = mild, 10 = unbearable)", "hint": "", "options": ["1-2 (Mild)", "3-4 (Moderate)", "5-6 (Significant)", "7-8 (Severe)", "9-10 (Unbearable)"]},
    {"key": "pattern", "question": "Is it constant or does it come and go?", "hint": "", "options": ["Constant, always there", "Comes and goes", "Getting worse over time", "Getting better", "Only happens sometimes"]},
    {"key": "other_symptoms", "question": "Are you experiencing any other symptoms alongside this?", "hint": "e.g. nausea, dizziness, fever, fatigue... or select None", "options": ["No other symptoms", "Nausea or vomiting", "Fever or chills", "Dizziness", "Fatigue or weakness", "Other (type below)"]},
    {"key": "known_conditions", "question": "Do you have any known medical conditions relevant to this symptom?", "hint": "This helps make triage safer and more accurate.", "options": ["No known conditions", "Diabetes", "Asthma", "High blood pressure", "Heart condition", "Pregnancy", "Other (type below)"]},
    {"key": "current_medications", "question": "Are you currently taking any medications?", "hint": "List known meds if relevant. You can type specific names below.", "options": ["No current medications", "Yes, regular medications", "Yes, as-needed medications", "Not sure", "Other (type below)"]},
    {"key": "red_flags", "question": "Any red-flag symptoms right now?", "hint": "Select any urgent danger signs if present.", "options": ["No red flags", "Chest pain or pressure", "Trouble breathing", "Fainting or confusion", "Severe bleeding", "One-sided weakness or slurred speech", "Other (type below)"]},
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
        "- Known conditions: " + assessment_data.get("known_conditions", "Not specified") + "\n"
        "- Current medications: " + assessment_data.get("current_medications", "Not specified") + "\n"
        "- Red-flag symptoms: " + assessment_data.get("red_flags", "Not specified") + "\n"
        "- Age group: " + assessment_data.get("age", "Not specified") + "\n"
        "- Biological sex: " + assessment_data.get("gender", "Not specified") + "\n\n"
        "INSTRUCTIONS:\n"
        "1. If the main symptom looks like a typo, interpret the most likely intended symptom.\n"
        "2. Consider all symptoms holistically.\n"
        "3. Provide realistic, evidence-based possible conditions.\n"
        "4. Urgency must reflect severity, red-flag symptoms, age, and comorbidity risk.\n"
        "5. If any red flags are present, escalate urgency and explicitly recommend urgent or emergency care.\n"
        "6. Do not overdiagnose, use cautious probability language.\n"
        + lang_note + "\n\n"
        "Medical research context:\n" + context + "\n\n"
        "Respond in EXACTLY this format:\n"
        "URGENCY: [one of: Self-care at home / See a doctor soon / Seek urgent care today / Go to emergency NOW]\n"
        "CONDITIONS: [condition 1] | [condition 2] | [condition 3]\n"
        "NEXT STEPS: [step 1] | [step 2] | [step 3]\n"
        "SUMMARY: [2-3 warm, clear sentences summarising the assessment]\n"
        "SAFETY: [one line with emergency fallback if symptoms worsen]\n"
    )
    r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1024)
    return r.choices[0].message.content

def parse_report(report_text):
    parsed = {"urgency": "", "conditions": [], "next_steps": [], "summary": "", "safety": ""}
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
        elif line.startswith("SAFETY:"):
            parsed["safety"] = line.replace("SAFETY:", "").strip()
SESSION_DEFAULTS = {
    "session_started": True,
    "messages": [],
    "qcount": 0,
    "feedback": {},
    "nav_clicked": False,
    "patient_memory": {"symptoms": [], "conditions": [], "medications": []},
    "uploader_key": 0,
    "last_pdf_context": "",
    "last_pdf_name": "",
    "last_image_context": "",
    "mode": "chat",
    "assessment_stage": 0,
    "assessment_data": {},
    "assessment_complete": False,
    "assessment_report": None,
    "assessment_parsed": None,
    "selected_language": "English",
    "patient_name": "",
    "emergency_detected": False,
    "emergency_reason": "",
    "triage_assessment": None,
    "last_sources": [],
    "eval_log": [],
    "response_times": [],
    "admin_authenticated": False,
    "admin_attempt_failed": False,
    "chat_input_key": 0,
    "is_authenticated": False,
    "is_guest": False,
    "user_email_hash": "",
    "user_email_display": "",
    "auth_error": "",
    "auth_view": "choose",
    "current_conversation_id": "",
    "pending_user_input": "",
    "rx_reader_result": None,
    "rx_uploader_key": 0,
    "home_show_vision_upload": False,
    "home_show_voice": False,
    "voice_audio_key": 0,
}

for _k, _v in SESSION_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (dict, list)) else _v


with st.sidebar:
    _brand_logo_uri = get_brand_logo_data_uri()
    if _brand_logo_uri:
        st.markdown(
            '<div class="md-logo-wrap md-logo-image-wrap">'
            '<img class="md-logo-image" src="' + _brand_logo_uri + '" alt="' + ui_escape(APP_TITLE) + '"/>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Fallback if the asset is missing — keep the Material Symbol mark + text.
        _logo_block = '<div class="md-logo-mark"><span class="material-symbols-rounded">medical_services</span></div>'
        st.markdown(
            '<div class="md-logo-wrap">'
            + _logo_block +
            '<div>'
            '<div class="md-logo-text">' + APP_TITLE + '</div>'
            '<div class="md-logo-sub">' + APP_SUBTITLE + '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── Primary nav ─────────────────────────────────────────────
    _mode = st.session_state.mode
    nav_items = [
        ("home", "Home", "chat", ":material/home:"),
        ("overview", "Health Overview", "overview", ":material/monitoring:"),
        ("symptom", "Symptoms Checker", "assessment", ":material/stethoscope:"),
        ("prescription", "Prescription Reader", "rx_reader", ":material/prescriptions:"),
        ("records", "Health Records", "records", ":material/lab_profile:"),
        ("meds", "Medications", "medications", ":material/pill:"),
        ("insights", "Ai Insights", "insights", ":material/auto_awesome:"),
        ("appts", "Appointments", "appointments", ":material/calendar_month:"),
    ]
    _nav_clicked = st.session_state.get("nav_clicked", False)
    for nav_key, nav_label, target_mode, nav_icon in nav_items:
        # Home and New Chat both target chat mode; differentiate by message presence
        # so exactly one is active at a time. Both also gated on nav_clicked so that
        # nothing is highlighted on a fresh load — the active state appears only
        # after the user explicitly navigates.
        if nav_key == "home":
            is_active = _nav_clicked and (_mode == "chat") and not st.session_state.messages
        elif nav_key == "new":
            is_active = _nav_clicked and (_mode == "chat") and bool(st.session_state.messages)
        else:
            is_active = (target_mode == _mode)
        _btn_type = "primary" if is_active else "secondary"
        active_cls = "md-nav-active" if is_active else ""
        st.markdown('<div class="' + active_cls + '">', unsafe_allow_html=True)
        if st.button(nav_label, key="nav_" + nav_key, use_container_width=True, icon=nav_icon, type=_btn_type):
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.session_state.nav_clicked = True
            if nav_key == "home" or nav_key == "new":
                start_new_chat_session()
            else:
                st.session_state.mode = target_mode
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Divider sits between Appointments (last nav item) and the profile chip.
    st.markdown("---")

    # ── Profile chip — sits where the Premium card used to. Live-updates with auth state. ──
    if st.session_state.is_authenticated:
        _profile_em = st.session_state.user_email_display or ""
        _saved_name = (st.session_state.patient_name or "").strip()
        if _saved_name and _saved_name.lower() != "guest":
            _profile_nm = _saved_name
        elif _profile_em and "@" in _profile_em:
            _profile_nm = _profile_em.split("@", 1)[0].replace(".", " ").replace("_", " ").title() or "Patient"
        else:
            _profile_nm = "Patient"
        _profile_in = (_profile_nm[0] if _profile_nm and _profile_nm != "Patient" else (_profile_em[0] if _profile_em else "P")).upper()
        _profile_sub = _profile_em if _profile_em else "View profile"
    else:
        _profile_nm = "Guest"
        _profile_in = "G"
        _profile_sub = "Sign in to save your data"
    # Sidebar CSS overrides for the native Streamlit buttons live in assets/style.css
    # (loaded once via _load_app_css).

_is_admin = st.session_state.admin_authenticated

# ── Patient Profile Auth Gate ────────────────────────────────────────
# Only enforced when Firebase is connected and user is not admin.
# Users can also continue as Guest (no persistence).
if (not _is_admin) and (not st.session_state.is_authenticated) and (not st.session_state.is_guest) and st.session_state.mode != "privacy":
    # Auth-page CSS lives in assets/style.css (loaded once via _load_app_css).
    _auth_shield_html = '<div class="md-auth-shield material-symbols-rounded">shield_person</div>'
    _auth_shield_uri = load_asset_data_uri("auth_welcome_shield.png")
    if _auth_shield_uri:
        _auth_shield_html = (
            '<div class="md-auth-shield md-auth-shield-image">'
            '<img src="' + _auth_shield_uri + '" alt="MediChat welcome shield icon">'
            '</div>'
        )

    st.markdown(
        '<div class="md-auth-welcome-card">'
        '<div class="md-auth-deco-dots"></div>'
        + _auth_shield_html +
        '<div class="md-auth-welcome-content">'
        '<div class="md-auth-welcome-title">Welcome to MediChat</div>'
        '<div class="md-auth-welcome-copy">Sign in to save your health profile across visits, or continue as a guest for a one-time chat.</div>'
        '<div class="md-auth-chip-row">'
        '<span class="md-auth-chip"><span class="material-symbols-rounded">lock</span>Private</span>'
        '<span class="md-auth-chip"><span class="material-symbols-rounded">person_check</span>Secure guest mode</span>'
        '<span class="md-auth-chip"><span class="material-symbols-rounded">verified</span>APP + HIPAA standard</span>'
        '<span class="md-auth-chip"><span class="material-symbols-rounded">verified_user</span>Health data protected</span>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    auth_main_col, auth_side_col = st.columns([2.35, 0.85], gap="large")

    with auth_main_col:
        view = st.session_state.auth_view
        if view == "choose":
            tab_signin, tab_signup, tab_guest = st.tabs(["Sign in", "Create profile", "Guest"])
            with tab_signin:
                with st.form("signin_form", clear_on_submit=False):
                    si_email = st.text_input("Email", placeholder="you@example.com", key="si_email", icon=":material/mail:")
                    si_pin = st.text_input("4-8 digit PIN", type="password", max_chars=8, key="si_pin", icon=":material/lock:")
                    st.markdown(
                        '<div class="md-auth-forgot-row">'
                        '<a class="md-auth-forgot-link" href="' + ui_escape(PRIVACY_POLICY_URL) + '" target="_blank">Forgot PIN?</a>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    si_btn = st.form_submit_button(
                        "Sign in",
                        key="auth_signin_submit",
                        use_container_width=True,
                        type="primary",
                        icon=":material/arrow_forward:"
                    )

                st.markdown('<div class="md-auth-signin-actions">', unsafe_allow_html=True)
                st.markdown('<div class="md-auth-or-divider"><span>or</span></div>', unsafe_allow_html=True)
                if st.button("Continue as Guest", key="go_guest_btn_signin", use_container_width=True, icon=":material/person:"):
                    st.session_state.is_guest = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(
                    '<div class="md-auth-meta md-auth-meta-solo">'
                    '<div class="md-auth-security-note">'
                    '<span class="material-symbols-rounded">shield_lock</span>'
                    '<span>We use secure, privacy-first authentication for your health information.</span>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                if si_btn:
                    if not si_email or "@" not in si_email or not si_pin or not si_pin.isdigit() or len(si_pin) < 4 or len(si_pin) > 8:
                        st.error("Please enter a valid email and a 4-8 digit numeric PIN.")
                    else:
                        profile, status = authenticate_profile(si_email, si_pin)
                        if status == "ok":
                            st.session_state.is_authenticated = True
                            st.session_state.user_email_hash = profile["email_hash"]
                            st.session_state.user_email_display = si_email.strip()
                            st.session_state.patient_name = profile.get("name", "") or ""
                            st.session_state.patient_memory = profile.get("patient_memory", {"symptoms": [], "conditions": [], "medications": []})
                            st.session_state.selected_language = profile.get("language", "English")
                            recent = list_conversations(profile["email_hash"], limit=1)
                            if recent:
                                conv = load_conversation(st.session_state.user_email_hash, recent[0]["id"])
                                if conv:
                                    st.session_state.current_conversation_id = recent[0]["id"]
                                    st.session_state.messages = conv.get("messages", []) or []
                                    st.session_state.qcount = sum(1 for m in st.session_state.messages if m.get("role") == "user")
                            else:
                                st.session_state.current_conversation_id = ""
                                st.session_state.messages = []
                                st.session_state.qcount = 0
                            st.success("Welcome back" + ((", " + profile.get("name", "")) if profile.get("name") else "") + ". Loading your profile…")
                            st.rerun()
                        elif status == "not_found":
                            st.error("No profile found for that email. Switch to 'Create profile' to start one.")
                        elif status == "wrong_pin":
                            st.error("Incorrect PIN. Try again or recover your account by creating a new profile with a different email.")

            with tab_signup:
                with st.form("signup_form", clear_on_submit=False):
                    su_email = st.text_input("Email", placeholder="you@example.com", key="su_email", icon=":material/mail:")
                    su_name = st.text_input("First name (optional)", key="su_name", max_chars=30, icon=":material/person:")
                    su_pin = st.text_input("Choose a 4-8 digit PIN", type="password", max_chars=8, key="su_pin", icon=":material/lock:")
                    su_pin2 = st.text_input("Confirm PIN", type="password", max_chars=8, key="su_pin2", icon=":material/password:")
                    su_btn = st.form_submit_button(
                        "Create profile",
                        key="auth_signup_submit",
                        use_container_width=True,
                        type="primary",
                        icon=":material/person_add:"
                    )
                if su_btn:
                    if not su_email or "@" not in su_email:
                        st.error("Please enter a valid email.")
                    elif not su_pin or not su_pin.isdigit() or len(su_pin) < 4 or len(su_pin) > 8:
                        st.error("PIN must be 4-8 digits, numbers only.")
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
                    '<div class="md-auth-guest-intro">'
                    '<div class="md-auth-guest-title">'
                    '<span class="material-symbols-rounded">visibility_off</span>'
                    'No account, no trace'
                    '</div>'
                    '<div class="md-auth-guest-copy">'
                    'Try MediChat without signing up. Your conversation lives only in this tab. Close it and everything is forgotten. '
                    'Switch to a profile any time for continuity across visits.'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="md-auth-signin-actions md-auth-signin-actions-guest">', unsafe_allow_html=True)
                if st.button("Continue as Guest", use_container_width=True, key="go_guest_btn", icon=":material/person:"):
                    st.session_state.is_guest = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="md-auth-privacy-foot">'
            'We store only irreversible email hashes and salted PIN hashes, designed to align with Australian Privacy Principles (APP), HIPAA-style safeguards, and Notifiable Data Breach commitments. '
            '<a href="' + ui_escape(PRIVACY_POLICY_URL) + '" target="_blank">Read the full Privacy Policy →</a>'
            '</div>',
            unsafe_allow_html=True
        )

    with auth_side_col:
        st.markdown(
            '<div class="md-auth-side-card">'
            '<h3 class="md-auth-side-title">Why sign in?</h3>'
            '<div class="md-auth-side-subline"></div>'
            '<div class="md-auth-benefit">'
            '<div class="md-auth-benefit-ic material-symbols-rounded">folder_managed</div>'
            '<div><p class="md-auth-benefit-title">Save your health history</p><div class="md-auth-benefit-copy">Keep your conversations, symptoms, and insights in one place.</div></div>'
            '</div>'
            '<div class="md-auth-benefit">'
            '<div class="md-auth-benefit-ic material-symbols-rounded">badge</div>'
            '<div><p class="md-auth-benefit-title">Access your Health Passport</p><div class="md-auth-benefit-copy">View and manage your verified health information anytime.</div></div>'
            '</div>'
            '<div class="md-auth-benefit">'
            '<div class="md-auth-benefit-ic material-symbols-rounded">sync</div>'
            '<div><p class="md-auth-benefit-title">Sync reports & medications</p><div class="md-auth-benefit-copy">Automatically sync your reports and medications across devices.</div></div>'
            '</div>'
            '<div class="md-auth-side-bottom">'
            '<span class="material-symbols-rounded">shield_locked</span>'
            '<span>Your health data is encrypted and protected at all times.</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
    st.stop()

if _is_admin:
    admin_c1, admin_c2 = st.columns([5, 1])
    with admin_c2:
        if st.button("🔒 Logout", key="admin_logout"):
            st.session_state.admin_authenticated = False
            st.session_state.mode = "chat"
            st.rerun()
    # Admin gets a compact mode switcher to access Analytics
    if st.session_state.mode != "eval":
        if st.button("📊 Open Admin Analytics", key="open_eval_btn"):
            st.session_state.mode = "eval"
            st.rerun()
else:
    if st.session_state.mode == "eval":
        st.session_state.mode = "chat"
    # Mode is now driven by the sidebar nav (Home / Symptoms Checker).

st.markdown('<div class="disclaimer">MediChat offers general health information, not personal medical advice. Please consult a qualified doctor for any health concerns that need diagnosis or treatment.</div>', unsafe_allow_html=True)

if st.session_state.emergency_detected:
    reason = st.session_state.get("emergency_reason", "Emergency indicators detected")
    st.markdown(
        '<div class="emergency-banner">'
        '<div class="emergency-title">🚨 This May Be a Medical Emergency</div>'
        '<div class="emergency-text"><strong>Detected pattern:</strong> ' + ui_text(reason, 120) + '. Based on what you described, you may need immediate medical attention. Please stop and call emergency services now. Do not wait.</div>'
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

# Triage tier banner removed per UI/UX refinement pass to clear screen real estate.

home_user_input = ""
home_submit = False
home_upload_clicked = False
home_voice_clicked = False
home_uploaded_image = None
home_vision_analyze = False
home_empty_chat = st.session_state.mode == "chat" and not st.session_state.messages


# ── Mode renderers (extracted from former if/elif st.session_state.mode chain) ──

def render_chat():
    # ── New Dashboard Home (only on empty chat) ─────────────────────
    if not st.session_state.messages:
        _local_now = get_user_local_now()
        _hour = _local_now.hour
        if 5 <= _hour < 12:
            _tod = "morning"
        elif 12 <= _hour < 17:
            _tod = "afternoon"
        elif 17 <= _hour < 21:
            _tod = "evening"
        else:
            _tod = "night"
        _disp_name = (st.session_state.patient_name if st.session_state.patient_name and st.session_state.patient_name != "Guest" else "")
        _greet = "Good " + _tod + (", " + _disp_name if _disp_name else "") + " 👋"
        st.markdown(
            '<div class="md-greet-wrap md-home-greet-wrap md-home-head-left">'
            '<div class="md-greet">' + ui_text(_greet, 80) + '</div>'
            '<div class="md-subgreet">How can I help you today?</div>'
            '</div>',
            unsafe_allow_html=True
        )

        home_main, home_side = st.columns([2.38, 0.95], gap="medium")

        with home_main:
            # Quick action cards
            qa_cols = st.columns(4, gap="small")
            qa_specs = [
                ("qa_headache", "Headache", "I have a headache and would like to understand what might be causing it.", ":material/psychology:"),
                ("qa_tired", "Low energy", "I have been feeling unusually tired lately. What could be the reason?", ":material/battery_low:"),
                ("qa_symptoms", "Check symptoms", "_route_assessment", ":material/monitor_heart:"),
                ("qa_sleep", "Better sleep", "Can you suggest ways to improve my sleep quality?", ":material/bedtime:"),
            ]
            for i, (qa_key, qa_label, qa_query, qa_icon) in enumerate(qa_specs):
                with qa_cols[i]:
                    if st.button(qa_label, key=qa_key, use_container_width=True, icon=qa_icon):
                        if qa_query == "_route_assessment":
                            st.session_state.mode = "assessment"
                            st.rerun()
                        else:
                            st.session_state.pending_user_input = qa_query
                            st.rerun()

            with st.form("home_chat_form", clear_on_submit=True):
                home_user_input = st.text_area(
                    "Start a chat",
                    placeholder="Ask anything about your health...",
                    label_visibility="collapsed",
                    height=100,
                    key="home_chat_input_" + str(st.session_state.chat_input_key),
                )
                ac1, ac2, ac_spacer, ac3 = st.columns([0.56, 0.56, 4.0, 0.56], vertical_alignment="center")
                with ac1:
                    home_upload_clicked = st.form_submit_button("Upload", key="home_upload_btn", icon=":material/attach_file:", use_container_width=True)
                with ac2:
                    home_voice_clicked = st.form_submit_button("Voice", key="home_voice_btn", icon=":material/mic:", use_container_width=True)
                with ac3:
                    home_submit = st.form_submit_button(" ", key="home_send_btn", icon=":material/send:", use_container_width=True, type="primary")
            st.markdown('<div class="md-composer-glow"></div>', unsafe_allow_html=True)
            if home_upload_clicked:
                st.session_state.home_show_vision_upload = True
                st.session_state.home_show_voice = False
                st.rerun()
            if home_voice_clicked:
                st.session_state.home_show_voice = True
                st.session_state.home_show_vision_upload = False
                st.rerun()

            if st.session_state.get("home_show_vision_upload", False):
                st.markdown(
                    '<div class="md-vision-panel">'
                    '<div class="md-panel-icon material-symbols-rounded">image_search</div>'
                    '<div><div class="md-panel-title">Vision Ai upload</div>'
                    '<div class="md-panel-subtitle">Upload an X-ray, scan photo, skin image, or blood report PDF for cautious visual/report analysis.</div></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                home_uploaded_image = st.file_uploader(
                    "Upload for Vision Ai",
                    type=["jpg", "jpeg", "png", "pdf"],
                    label_visibility="collapsed",
                    key="home_vision_uploader_" + str(st.session_state.uploader_key),
                    help="Upload a medical image or PDF report for Vision Ai analysis.",
                )
                if home_uploaded_image:
                    is_home_pdf = home_uploaded_image.name.lower().endswith(".pdf")
                    if is_home_pdf:
                        st.markdown(
                            '<div class="md-file-preview md-home-file-preview">'
                            '<div class="md-file-icon">PDF</div>'
                            '<div style="flex:1;min-width:0;">'
                            '<div class="md-file-name">' + ui_text(home_uploaded_image.name, 90) + '</div>'
                            '<div class="md-file-help">Ready for Vision Ai report review.</div>'
                            '</div>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.image(home_uploaded_image, caption="Ready for Vision Ai analysis", use_column_width=True)
                vu1, vu2 = st.columns([1, 1])
                with vu1:
                    home_vision_analyze = st.button("Analyze with Vision Ai", key="home_vision_analyze", type="primary", use_container_width=True, icon=":material/image_search:")
                    if home_vision_analyze and not home_uploaded_image:
                        st.error("Please upload an image or PDF first.")
                        home_vision_analyze = False
                with vu2:
                    if st.button("Cancel upload", key="home_vision_cancel", use_container_width=True, icon=":material/close:"):
                        st.session_state.home_show_vision_upload = False
                        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
                        st.rerun()

            if st.session_state.get("home_show_voice", False):
                st.markdown(
                    '<div class="md-voice-panel">'
                    '<div class="md-panel-icon material-symbols-rounded">mic</div>'
                    '<div><div class="md-panel-title">Voice note</div>'
                    '<div class="md-panel-subtitle">Record or upload a short voice question. MediChat will transcribe it before answering.</div></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                voice_audio = None
                if hasattr(st, "audio_input"):
                    voice_audio = st.audio_input("Record your question", key="voice_audio_" + str(st.session_state.voice_audio_key))
                else:
                    voice_audio = st.file_uploader(
                        "Upload a voice note",
                        type=["wav", "mp3", "m4a", "ogg", "webm"],
                        key="voice_upload_" + str(st.session_state.voice_audio_key),
                        help="Your Streamlit version does not expose microphone recording, so upload an audio note instead.",
                    )
                vc1, vc2 = st.columns([1, 1])
                with vc1:
                    if st.button("Transcribe voice", key="voice_transcribe", type="primary", use_container_width=True, icon=":material/graphic_eq:"):
                        transcript = transcribe_voice_note(voice_audio)
                        if transcript:
                            st.session_state.pending_user_input = transcript
                            st.session_state.home_show_voice = False
                            st.session_state.voice_audio_key = st.session_state.get("voice_audio_key", 0) + 1
                            st.rerun()
                        else:
                            st.error("I could not transcribe that audio. Please try a clearer recording or type the question.")
                with vc2:
                    if st.button("Cancel voice", key="voice_cancel", use_container_width=True, icon=":material/close:"):
                        st.session_state.home_show_voice = False
                        st.session_state.voice_audio_key = st.session_state.get("voice_audio_key", 0) + 1
                        st.rerun()

            st.markdown(
                '<div class="md-home-composer-note"><span class="material-symbols-rounded md-disclaimer-shield">verified_user</span>MediChat Ai can make mistakes. Please consult a healthcare professional for medical advice.</div>',
                unsafe_allow_html=True
            )

            # ── Smart Actions (available for all users) ───────────────
            st.markdown('<div class="md-smart-head"><div class="md-smart-title">Smart Actions</div></div>', unsafe_allow_html=True)
            # Markdown bold (**...**) on the title makes it bold even when
            # it wraps to 2 lines (unlike CSS ::first-line which only styles
            # the first VISUAL line). The two trailing spaces before \n
            # render a line break in markdown, separating title + subtitle.
            sa_specs = [
                ("sa_sym", "**Vision Ai**  \nAnalyze images and X-rays", "vision", ":material/image_search:", "md-smart-purple"),
                ("sa_rec", "**Health Records**  \nUpload medical reports", "records", ":material/medical_information:", "md-smart-green"),
                ("sa_ins", "**Prescription Reader**  \nScan prescriptions", "rx_reader", ":material/prescriptions:", "md-smart-pink"),
                ("sa_appt", "**Appointments**  \nManage appointments", "appointments", ":material/calendar_month:", "md-smart-blue"),
            ]
            sa_cols = st.columns(4, gap="small")
            for i, (sk, label, action, icon_name, accent_cls) in enumerate(sa_specs):
                with sa_cols[i]:
                    if st.button(label, key=sk, use_container_width=True, icon=icon_name):
                        if action == "vision":
                            st.session_state.mode = "chat"
                            st.session_state.home_show_voice = False
                            st.session_state.home_show_vision_upload = True
                        else:
                            st.session_state.mode = action
                        st.rerun()

            # ── Daily Health Tip carousel (4 live-data slides, 3s rotation) ──
            _tip_daily = get_daily_metrics()
            _tip_water_raw = _tip_daily.get("water_glasses")
            _tip_sleep_raw = _tip_daily.get("sleep_hours")
            _tip_steps_raw = _tip_daily.get("steps")
            _tip_hr_raw = _tip_daily.get("heart_rate_resting")
            _tip_water_metric = (str(int(_tip_water_raw)) + " / 8 glasses today") if _tip_water_raw is not None else "Not logged today. Tap Health Overview to log"
            _tip_sleep_metric = ("Last night: " + ("%.1f" % float(_tip_sleep_raw)) + "h") if _tip_sleep_raw else "Sleep not logged. Log it on Health Overview"
            _tip_steps_metric = (format(int(_tip_steps_raw), ",") + " / 10,000 steps") if _tip_steps_raw else "Steps not logged. Log them on Health Overview"
            _tip_hr_metric = ("Resting HR " + str(int(_tip_hr_raw)) + " BPM") if _tip_hr_raw else "Heart rate not logged. Log it on Health Overview"
            _tip_glass_svg = (
                '<svg viewBox="0 0 86 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
                '<defs><linearGradient id="mdGlassC" x1="0" y1="0" x2="0" y2="1">'
                '<stop offset="0%" stop-color="#bfdbfe" stop-opacity="0.55"/>'
                '<stop offset="100%" stop-color="#60a5fa" stop-opacity="0.85"/>'
                '</linearGradient></defs>'
                '<ellipse cx="43" cy="92" rx="30" ry="4" fill="#1e3a8a" opacity="0.08"/>'
                '<path d="M22 30 L64 30 L60 88 Q60 92 56 92 L30 92 Q26 92 26 88 Z" fill="url(#mdGlassC)" stroke="#3b82f6" stroke-width="1.6" stroke-linejoin="round"/>'
                '<ellipse cx="43" cy="30" rx="21" ry="3.4" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.4"/>'
                '<path d="M30 42 Q34 46 32 52 Q30 56 33 60" stroke="#ffffff" stroke-width="1.8" stroke-linecap="round" fill="none" opacity="0.85"/>'
                '<circle cx="48" cy="58" r="2" fill="#ffffff" opacity="0.7"/>'
                '<circle cx="38" cy="72" r="1.4" fill="#ffffff" opacity="0.6"/>'
                '</svg>'
            )
            _tip_html = (
                '<div class="md-tip-carousel">'
                '<div class="md-tip-slide md-tip-water">'
                '<div>'
                '<div class="md-tip-eyebrow">Hydration tip</div>'
                '<div class="md-tip-title">Stay hydrated</div>'
                '<div class="md-tip-desc">Drinking enough water helps maintain energy and supports overall health.</div>'
                '<div class="md-tip-metric"><span class="material-symbols-rounded">water_drop</span>' + _tip_water_metric + '</div>'
                '</div>'
                '<div class="md-tip-illust md-tip-illust-svg">' + _tip_glass_svg + '</div>'
                '</div>'
                '<div class="md-tip-slide md-tip-sleep">'
                '<div>'
                '<div class="md-tip-eyebrow">Sleep insight</div>'
                '<div class="md-tip-title">Wind down earlier</div>'
                '<div class="md-tip-desc">Quality sleep boosts recovery, mood, and immunity. Aim for 7–9 hours every night.</div>'
                '<div class="md-tip-metric"><span class="material-symbols-rounded">bedtime</span>' + _tip_sleep_metric + '</div>'
                '</div>'
                '<div class="md-tip-illust"><span class="material-symbols-rounded">bedtime</span></div>'
                '</div>'
                '<div class="md-tip-slide md-tip-move">'
                '<div>'
                '<div class="md-tip-eyebrow">Movement</div>'
                '<div class="md-tip-title">Keep moving</div>'
                '<div class="md-tip-desc">A short walk after meals helps regulate blood sugar and energy levels through the day.</div>'
                '<div class="md-tip-metric"><span class="material-symbols-rounded">directions_walk</span>' + _tip_steps_metric + '</div>'
                '</div>'
                '<div class="md-tip-illust"><span class="material-symbols-rounded">directions_walk</span></div>'
                '</div>'
                '<div class="md-tip-slide md-tip-vitals">'
                '<div>'
                '<div class="md-tip-eyebrow">Vitals check</div>'
                '<div class="md-tip-title">Heart in good rhythm</div>'
                '<div class="md-tip-desc">A resting heart rate of 60–100 BPM is typical for healthy adults. Keep moving and resting well.</div>'
                '<div class="md-tip-metric"><span class="material-symbols-rounded">favorite</span>' + _tip_hr_metric + '</div>'
                '</div>'
                '<div class="md-tip-illust"><span class="material-symbols-rounded">favorite</span></div>'
                '</div>'
                '<div class="md-tip-indicators">'
                '<span class="md-tip-dot dot-1"></span>'
                '<span class="md-tip-dot dot-2"></span>'
                '<span class="md-tip-dot dot-3"></span>'
                '<span class="md-tip-dot dot-4"></span>'
                '</div>'
                '</div>'
            )
            st.markdown(_tip_html, unsafe_allow_html=True)

        with home_side:
            # ── Profile Snapshot ────────────────────────────────────────
                _mem_now = st.session_state.patient_memory or {}
                _cond_n = len(_mem_now.get("conditions") or [])
                _med_n = len(_mem_now.get("medications") or [])
                _sym_n = len(_mem_now.get("symptoms") or [])
                _convs_total = 0
                _last_visit_str = "-"
                if st.session_state.is_authenticated and st.session_state.user_email_hash:
                    _all_convs = list_conversations(st.session_state.user_email_hash, limit=100)
                    _convs_total = len(_all_convs)
                    if _all_convs and _all_convs[0].get("last_updated"):
                        try:
                            _lu = _all_convs[0]["last_updated"]
                            _delta = datetime.now(timezone.utc) - (_lu if _lu.tzinfo else _lu.replace(tzinfo=timezone.utc))
                            _h = int(_delta.total_seconds() // 3600)
                            _last_visit_str = (str(int(_delta.total_seconds() // 60)) + "m ago") if _h < 1 else (str(_h) + "h ago" if _h < 24 else str(_h // 24) + "d ago")
                        except Exception:
                            _last_visit_str = "recently"

                _snap_title = "Health Overview"
                _daily = get_daily_metrics()
                # Real values only — no fake fallbacks. Missing → "-" with no status pill.
                _hr = _daily.get("heart_rate_resting")
                _steps_val = _daily.get("steps")
                _sleep_hours = _daily.get("sleep_hours")
                _water_count = _daily.get("water_glasses")

                _heart_rate_display = (str(int(_hr)) + " BPM") if _hr else "-"
                _hr_status, _hr_cls = heart_rate_status(_hr)
                _heart_rate_status_text = _hr_status or ""

                _steps_display = (f"{int(_steps_val):,} / 10,000") if _steps_val else "-"
                _steps_status_text = (str(min(100, int(round((int(_steps_val) / 10000) * 100)))) + "%") if _steps_val else ""

                _sleep_display = (f"{float(_sleep_hours):.1f}h") if _sleep_hours else "-"
                _sl_status, _sl_cls = sleep_status(_sleep_hours)
                _sleep_status_text = _sl_status or ""

                _wc_int = int(_water_count) if _water_count is not None else None
                _water_display = (f"{_wc_int} / 8 glasses") if _wc_int is not None else "-"
                _water_status_text = (str(min(100, int(round((_wc_int / 8) * 100)))) + "%") if _wc_int else ""

                _tiles = [
                    ("md-accent-pink", "favorite", "Heart Rate", _heart_rate_display, _heart_rate_status_text, "md-line-pink"),
                    ("md-accent-green", "directions_walk", "Steps", _steps_display, _steps_status_text, "md-line-green"),
                    ("md-accent-purple", "bedtime", "Sleep", _sleep_display, _sleep_status_text, "md-line-purple"),
                    ("md-accent-blue", "water_drop", "Water Intake", _water_display, _water_status_text, "md-line-blue"),
                ]
                snap_html = (
                    '<div class="md-rcard md-snap-card">'
                    '<div class="md-rcard-head"><div class="md-rcard-title">' + _snap_title + '</div><span class="md-rcard-link md-rcard-link-btn" style="cursor: default;">See all</span></div>'
                    '<div class="md-snap-grid">'
                )
                for _cls, _icon, _lbl, _val, _status, _line_cls in _tiles:
                    snap_html += (
                        '<div class="md-snap-tile">'
                        '<div class="md-snap-icon ' + _cls + ' material-symbols-rounded">' + ui_escape(_icon) + '</div>'
                        '<div class="md-snap-text">'
                        '<div><div class="md-snap-label">' + ui_text(_lbl, 40) + '</div>'
                        '<div class="md-snap-value">' + ui_text(_val, 40) + '</div></div>'
                        '<div class="md-snap-status">' + ui_text(_status, 20) + '</div>'
                        '</div>'
                        '<svg class="md-spark ' + _line_cls + '" viewBox="0 0 96 28" aria-hidden="true"><path d="M2 18 C12 18 14 11 24 14 S38 24 48 15 S62 2 72 10 S84 19 94 13" /></svg>'
                        '</div>'
                    )
                snap_html += '</div></div>'
                st.markdown(snap_html, unsafe_allow_html=True)
                st.markdown('<div class="md-rail-link-btn">', unsafe_allow_html=True)
                if st.button("See all health data →", key="home_overview_see_all", use_container_width=True):
                    st.session_state.mode = "overview"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

                # Recent Conversations (real, from Firestore)
                recent_html = '<div class="md-rcard md-rcard-recent"><div class="md-rcard-head"><div class="md-rcard-title">Recent Conversations</div><span class="md-rcard-link md-rcard-link-btn" style="cursor: default;">See all</span></div>'
                if st.session_state.is_authenticated and st.session_state.user_email_hash:
                    _recent = list_conversations(st.session_state.user_email_hash, limit=4)
                else:
                    _recent = []  # Real conversations only: no mock entries.
                if _recent:
                    for _r in _recent:
                        _rt = (_r.get("title") or "Chat")[:32]
                        _ru = _r.get("last_updated")
                        try:
                            if _ru and hasattr(_ru, "strftime"):
                                _delta = datetime.now(timezone.utc) - (_ru if _ru.tzinfo else _ru.replace(tzinfo=timezone.utc))
                                _hours = int(_delta.total_seconds() // 3600)
                                if _hours < 1:
                                    _ago = str(int(_delta.total_seconds() // 60)) + "m ago"
                                elif _hours < 24:
                                    _ago = str(_hours) + "h ago"
                                else:
                                    _ago = str(_hours // 24) + "d ago"
                            else:
                                _ago = ""
                        except Exception:
                            _ago = ""
                        recent_html += '<div class="md-conv-row"><div class="md-conv-bubble material-symbols-rounded">chat_bubble</div><div class="md-conv-title">' + ui_text(_rt, 40) + '</div><div class="md-conv-time">' + ui_text(_ago, 20) + '</div></div>'
                else:
                    if st.session_state.is_authenticated:
                        recent_html += '<div class="md-conv-row md-conv-empty">No recent conversations yet. Start a chat to see it here.</div>'
                    else:
                        recent_html += '<div class="md-conv-row md-conv-empty">Sign in to save and revisit your conversations.</div>'
                recent_html += '</div>'
                st.markdown(recent_html, unsafe_allow_html=True)
                st.markdown('<div class="md-view-all-wrap">', unsafe_allow_html=True)
                if st.button("View all chats →", key="view_all_recent", use_container_width=True):
                    st.session_state.mode = "history"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

                # Live Health Passport (fills empty right-rail area with a real, usable feature).
                _records_total = len(list_health_records())
                _meds_total = len(list_medications())
                _appts_total = len(list_appointments())
                _vitals_logged = bool(_hr or _steps_val or _sleep_hours or (_water_count is not None and int(_water_count) > 0))
                _has_history = bool(_convs_total) if st.session_state.is_authenticated else bool(st.session_state.messages)
                _profile_ready = bool(st.session_state.is_authenticated and st.session_state.user_email_hash)
                _passport_checks = [
                    ("Profile linked", _profile_ready),
                    ("Chat history", _has_history),
                    ("Vitals logged", _vitals_logged),
                    ("Records uploaded", _records_total > 0),
                    ("Medications synced", _meds_total > 0),
                    ("Appointments tracked", _appts_total > 0),
                ]
                _passport_done = sum(1 for _, _ok in _passport_checks if _ok)
                _passport_total = len(_passport_checks)
                _passport_pct = int(round((_passport_done / _passport_total) * 100)) if _passport_total else 0

                _passport_html = (
                    '<div class="md-rcard md-passport-card">'
                    '<div class="md-passport-head">'
                    '<div class="md-passport-title-wrap"><span class="material-symbols-rounded">badge</span><div class="md-passport-title">Health Passport</div></div>'
                    '<div class="md-passport-pct">' + str(_passport_pct) + '% ready</div>'
                    '</div>'
                    '<div class="md-passport-sub">Live profile completeness across vitals, records, medications, and appointments.</div>'
                    '<div class="md-passport-progress"><div class="md-passport-fill" style="width:' + str(_passport_pct) + '%;"></div></div>'
                )
                for _label, _ok in _passport_checks:
                    _row_cls = "ok" if _ok else "todo"
                    _icon = "check_circle" if _ok else "radio_button_unchecked"
                    _status = "Complete" if _ok else "Pending"
                    _passport_html += (
                        '<div class="md-passport-check ' + _row_cls + '">'
                        '<div class="md-passport-check-left">'
                        '<span class="material-symbols-rounded">' + _icon + '</span>'
                        '<span class="md-passport-check-label">' + ui_text(_label, 40) + '</span>'
                        '</div>'
                        '<span class="md-passport-status ' + _row_cls + '">' + _status + '</span>'
                        '</div>'
                    )
                _passport_html += '</div>'
                st.markdown(_passport_html, unsafe_allow_html=True)

                _pp1, _pp2 = st.columns(2, gap="small")
                with _pp1:
                    if st.button("Open Records", key="home_passport_records", use_container_width=True, icon=":material/folder_managed:"):
                        st.session_state.mode = "records"
                        st.rerun()
                with _pp2:
                    if st.button("Update Vitals", key="home_passport_overview", use_container_width=True, icon=":material/monitoring:"):
                        st.session_state.mode = "overview"
                        st.rerun()
                if st.button("Sync Medications", key="home_passport_meds", use_container_width=True, icon=":material/pill:"):
                    st.session_state.mode = "medications"
                    st.rerun()
                # Daily Health Tip carousel is rendered below Smart Actions in home_main.



    show_hero = False

    if show_hero:
        st.markdown(
            '<div class="md-name-card">'
            '<div class="md-name-title">Personalise the conversation</div>'
            '<div class="md-name-subtitle">Add a first name if you want MediChat to greet you more naturally. You can skip this and start chatting right away.</div>'
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
        pass

    else:
        user_initial = "U"
        if st.session_state.patient_name and st.session_state.patient_name != "Guest":
            user_initial = st.session_state.patient_name[0].upper()
        user_name_label = st.session_state.patient_name if st.session_state.patient_name and st.session_state.patient_name != "Guest" else "You"

        # Page hero at top of every chat conversation (matches the design
        # mockup: title + subtitle + privacy indicator).
        st.markdown(
            '<div class="md-chat-hero">'
            '<div class="md-chat-hero-text">'
            '<div class="md-chat-hero-title">AI Health Conversation '
            # Inline SVG shield — zero font dependency, renders identically
            # across all browsers as a proper filled shield silhouette with
            # a soft check mark inside. Replaces the Material Symbols
            # `shield` glyph which was rendering ambiguously at small sizes.
            '<svg class="md-chat-hero-shield" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
            '<path d="M12 2 3.5 5.5v6c0 5.2 3.6 9.9 8.5 11 4.9-1.1 8.5-5.8 8.5-11v-6L12 2z"/>'
            '<path d="M10.6 14.4 8 11.8l-1.1 1.1 3.7 3.7 7-7-1.1-1.1-5.9 5.9z" fill="#ffffff"/>'
            '</svg></div>'
            '<div class="md-chat-hero-sub">Private, supportive guidance for your symptoms</div>'
            '</div>'
            '<div class="md-chat-hero-privacy">'
            '<span class="material-symbols-rounded">lock</span>'
            'Your conversation is private'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # MediChat brand logo loaded once for bot avatars across the conversation.
        _bot_avatar_uri = get_brand_logo_data_uri()
        _bot_avatar_html = (
            '<div class="av av-bot av-bot-image"><img src="' + _bot_avatar_uri + '" alt="MediChat AI"></div>'
            if _bot_avatar_uri
            else '<div class="av av-bot">M</div>'
        )

        for msg in st.session_state.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            msg_type = msg.get("type", "text")
            msg_ts = msg.get("ts", "")
            ts_html = ('<div class="bubble-ts">' + ui_escape(msg_ts) + '</div>') if msg_ts else ""
            if role == "user":
                safe_content = ui_lines(content)
                safe_initial = ui_text(user_initial, 2)
                if msg_type == "image":
                    st.markdown('<span class="image-tag">Medical image uploaded for analysis</span>', unsafe_allow_html=True)
                if content or msg_type != "image":
                    st.markdown(
                        '<div class="user-wrap">'
                        '<div class="user-stack">'
                        '<div class="user-bubble">' + safe_content + '</div>'
                        + ts_html +
                        '</div>'
                        '<div class="av av-user"><span class="material-symbols-rounded" style="font-size: 1.25rem;">person</span></div>'
                        '</div>',
                        unsafe_allow_html=True
                    )
            else:
                # Bot message — logo avatar + bubble with inline "MediChat AI"
                # header (with sparkle), the response text, and a right-aligned
                # timestamp under the bubble.
                st.markdown(
                    '<div class="bot-wrap">'
                    + _bot_avatar_html +
                    '<div class="bot-stack">'
                    '<div class="bot-bubble">'
                    '<div class="bot-bubble-head">'
                    '<span class="bot-bubble-name">MediChat AI</span>'
                    '<span class="bot-bubble-spark material-symbols-rounded">auto_awesome</span>'
                    '</div>'
                    + markdown_to_html(content) +
                    '</div>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                engine_used = msg.get("engine", "")
                msg_sources = msg.get("sources", [])
                is_image_response = "Image Analysis" in msg_sources or engine_used == "groq-vision"
                is_rx_response = "Prescription Reader" in msg_sources or str(engine_used).startswith("rx-reader")
                is_pdf_response = "PDF Report Analysis" in msg_sources

                engine_html = ""
                if str(engine_used).startswith("rx-reader"):
                    engine_html = '<span class="engine-badge engine-vision">Prescription Reader</span>'
                if is_rx_response:
                    source_tags = '<span class="source-tag">📝 Prescription Reader</span>'
                elif is_image_response:
                    source_tags = '<span class="source-tag">📷 Image Analysis</span>'
                elif is_pdf_response:
                    source_tags = '<span class="source-tag">📄 PDF Report Analysis</span>'
                else:
                    source_tags = "".join(['<span class="source-tag">📚 ' + ui_text(s, 50) + '</span>' for s in msg_sources])

                conf_html = ""
                conf_level = msg.get("confidence")
                conf_pct = msg.get("confidence_pct")
                if conf_level and conf_pct and not is_image_response and not is_rx_response and engine_used != "system":
                    conf_level = conf_level if conf_level in ("high", "medium", "low") else "low"
                    conf_label = {"high": "High Confidence", "medium": "Medium Confidence", "low": "Low Confidence"}.get(conf_level, "")
                    conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(conf_level, "#64748b")
                    conf_html = (
                        '<span class="confidence-pill conf-' + conf_level + '">' + conf_label + '</span>'
                        '<span class="confidence-bar"><span class="confidence-fill" style="width:' + str(conf_pct) + '%;background:' + conf_color + ';"></span></span>'
                        '<span class="rag-text">' + str(conf_pct) + '% match</span>'
                    )

                if engine_html or source_tags or conf_html:
                    parts = []
                    if engine_html or source_tags:
                        parts.append('<span class="meta-label meta-label-sources"><span class="material-symbols-rounded">verified_user</span>Sources</span>' + engine_html + source_tags)
                    if conf_html:
                        parts.append('<span class="meta-label meta-label-conf">Confidence</span>' + conf_html)
                    st.markdown('<div class="meta-row">' + "".join(parts) + '</div>', unsafe_allow_html=True)

                if msg_ts:
                    st.markdown('<div class="bot-ts">' + ui_escape(msg_ts) + '</div>', unsafe_allow_html=True)

    if st.session_state.messages:
        # Feedback row removed per user request.
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
    user_input = home_user_input
    submit = home_submit or home_vision_analyze
    uploaded_image = home_uploaded_image
    clear = False
    if not home_empty_chat:
        uploaded_image = None
        chat_upload_clicked = False
        chat_voice_clicked = False
        chat_clear_clicked = False
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your message",
                placeholder="Ask anything about your health...",
                label_visibility="collapsed",
                height=100,
                key="chat_input_" + str(st.session_state.chat_input_key),
            )
            # Action row: Upload + Voice + Clear pills on the left, big spacer,
            # round Send ball on the right. Clear lives inside the form so
            # it's visually grouped with the chat box (no orphan button below).
            fc1, fc2, fc3, fc_spacer, fc4 = st.columns([0.56, 0.56, 0.56, 5.2, 0.56], vertical_alignment="center")
            with fc1:
                chat_upload_clicked = st.form_submit_button("Upload", key="chat_upload_btn", icon=":material/attach_file:", use_container_width=True)
            with fc2:
                chat_voice_clicked = st.form_submit_button("Voice", key="chat_voice_btn", icon=":material/mic:", use_container_width=True)
            with fc3:
                chat_clear_clicked = st.form_submit_button("Clear", key="chat_clear_btn", icon=":material/delete:", use_container_width=True)
            with fc4:
                submit = st.form_submit_button(" ", key="chat_send_btn", icon=":material/send:", use_container_width=True, type="primary")
        # If user hit Clear inside the chat form, wire it to the same clear
        # action as the standalone button used by vision/voice panels.
        if chat_clear_clicked:
            clear = True

        if chat_upload_clicked:
            st.session_state.home_show_vision_upload = True
            st.session_state.home_show_voice = False
            st.rerun()
        if chat_voice_clicked:
            st.session_state.home_show_voice = True
            st.session_state.home_show_vision_upload = False
            st.rerun()

        if st.session_state.get("home_show_vision_upload", False):
            st.markdown(
                '<div class="md-vision-panel">'
                '<div class="md-panel-icon material-symbols-rounded">image_search</div>'
                '<div><div class="md-panel-title">Vision Ai upload</div>'
                '<div class="md-panel-subtitle">Upload an X-ray, scan photo, skin image, or blood report PDF for cautious visual/report analysis.</div></div>'
                '</div>',
                unsafe_allow_html=True
            )
            uploaded_image = st.file_uploader(
                "Upload for Vision Ai",
                type=["jpg", "jpeg", "png", "pdf"],
                label_visibility="collapsed",
                key="chat_vision_uploader_" + str(st.session_state.uploader_key),
                help="Upload a medical image or PDF report for Vision Ai analysis.",
            )
            if uploaded_image:
                is_chat_pdf = uploaded_image.name.lower().endswith(".pdf")
                if is_chat_pdf:
                    st.markdown(
                        '<div class="md-file-preview md-home-file-preview">'
                        '<div class="md-file-icon">PDF</div>'
                        '<div style="flex:1;min-width:0;">'
                        '<div class="md-file-name">' + ui_text(uploaded_image.name, 90) + '</div>'
                        '<div class="md-file-help">Ready for Vision Ai report review.</div>'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.image(uploaded_image, caption="Ready for Vision Ai analysis", use_column_width=True)
            vu1, vu2, vu3 = st.columns([1, 1, 1])
            with vu1:
                home_vision_analyze = st.button("Analyze with Vision Ai", key="chat_vision_analyze", type="primary", use_container_width=True, icon=":material/image_search:")
                submit = submit or home_vision_analyze
                if home_vision_analyze and not uploaded_image:
                    st.error("Please upload an image or PDF first.")
                    submit = False
            with vu2:
                if st.button("Cancel upload", key="chat_vision_cancel", use_container_width=True, icon=":material/close:"):
                    st.session_state.home_show_vision_upload = False
                    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
                    st.rerun()
            with vu3:
                clear = st.button("Clear chat", use_container_width=True, key="main_clear_btn", icon=":material/delete:")

        if st.session_state.get("home_show_voice", False):
            st.markdown(
                '<div class="md-voice-panel">'
                '<div class="md-panel-icon material-symbols-rounded">mic</div>'
                '<div><div class="md-panel-title">Voice note</div>'
                '<div class="md-panel-subtitle">Record or upload a short voice question. MediChat will transcribe it before answering.</div></div>'
                '</div>',
                unsafe_allow_html=True
            )
            voice_audio = None
            if hasattr(st, "audio_input"):
                voice_audio = st.audio_input("Record your question", key="voice_audio_chat_" + str(st.session_state.voice_audio_key))
            else:
                voice_audio = st.file_uploader(
                    "Upload a voice note",
                    type=["wav", "mp3", "m4a", "ogg", "webm"],
                    key="voice_upload_chat_" + str(st.session_state.voice_audio_key),
                    help="Your Streamlit version does not expose microphone recording, so upload an audio note instead.",
                )
            vc1, vc2, vc3 = st.columns([1, 1, 1])
            with vc1:
                if st.button("Transcribe voice", key="voice_transcribe_chat", type="primary", use_container_width=True, icon=":material/graphic_eq:"):
                    transcript = transcribe_voice_note(voice_audio)
                    if transcript:
                        st.session_state.pending_user_input = transcript
                        st.session_state.home_show_voice = False
                        st.session_state.voice_audio_key = st.session_state.get("voice_audio_key", 0) + 1
                        st.rerun()
                    else:
                        st.error("I could not transcribe that audio. Please try a clearer recording or type the question.")
            with vc2:
                if st.button("Cancel voice", key="voice_cancel_chat", use_container_width=True, icon=":material/close:"):
                    st.session_state.home_show_voice = False
                    st.session_state.voice_audio_key = st.session_state.get("voice_audio_key", 0) + 1
                    st.rerun()
            with vc3:
                clear = st.button("Clear chat", use_container_width=True, key="main_clear_btn_voice", icon=":material/delete:")

        # Plain-state Clear button removed — it's now an inline chip inside
        # the chat composer form (chat_clear_btn). Vision/voice panels still
        # have their own Clear next to Cancel above.

    # ── File preview (shown below button row once attached) ─────────────
    if uploaded_image and not home_empty_chat and not st.session_state.get("home_show_vision_upload", False):
        is_pdf_upload = uploaded_image.name.lower().endswith(".pdf")
        if is_pdf_upload:
            st.markdown(
                '<div class="md-file-preview">'
                '<div class="md-file-icon">PDF</div>'
                '<div style="flex:1;min-width:0;">'
                '<div class="md-file-name">' + ui_text(uploaded_image.name, 90) + '</div>'
                '<div class="md-file-help">Ready for analysis. Ask MediChat what you want to know about this report.</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            ia, ib, ic = st.columns([1, 2, 1])
            with ib:
                st.image(uploaded_image, caption="Ready for analysis", use_column_width=True)

    if not home_empty_chat:
        st.markdown(
            '<div class="md-home-composer-note">'
            '<span class="material-symbols-rounded md-disclaimer-shield">lock</span>'
            'This is not emergency care. If you feel seriously unwell, seek immediate medical attention.'
            '</div>',
            unsafe_allow_html=True
        )
    st.markdown('<div id="page-bottom-anchor" style="height:1px;"></div>', unsafe_allow_html=True)

    # Enter-to-send (Shift+Enter keeps newline) for both home and in-chat composers.
    st.markdown(
        '''<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" onload="(function(){try{if(!window.__medichatEnterSendBoundV4){window.__medichatEnterSendBoundV4=true;console.log('Enter-to-send V4 listener initialized');document.addEventListener('keydown',function(ev){if(!ev.defaultPrevented&&ev.key==='Enter'&&!ev.shiftKey&&!ev.ctrlKey&&!ev.metaKey&&!ev.altKey&&!ev.isComposing){const target=ev.target;if(target&&target.tagName==='TEXTAREA'){const label=(target.getAttribute('aria-label')||'').trim().toLowerCase();const placeholder=(target.getAttribute('placeholder')||'').trim().toLowerCase();if(label.includes('start a chat')||label.includes('your message')||placeholder.includes('ask anything about your health')||target.closest('[class*=st-key-home_chat_input], [class*=st-key-chat_input]')){ev.preventDefault();const form=target.closest('[data-testid=stForm]');if(form){const btn=form.querySelector('[class*=st-key-home_send_btn] button, [class*=st-key-chat_send_btn] button, button[kind=primaryFormSubmit], button[kind=primary], [data-testid=stFormSubmitButton] button');if(btn&&!btn.disabled)btn.click();}}}}},true);}}catch(e){console.error('Enter-to-send V4 error:',e);}})()" style="display:none;position:absolute;width:0;height:0;">''',
        unsafe_allow_html=True,
    )

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
        # Start a fresh chat. The existing conversation stays saved in history.
        st.session_state.last_pdf_name = ""
        st.session_state.emergency_reason = ""
        start_new_chat_session()
        st.rerun()

    pending_user_input = st.session_state.get("pending_user_input", "")
    effective_user_input = pending_user_input if pending_user_input else user_input
    auto_submit = bool(pending_user_input)

    if (submit or auto_submit) and (effective_user_input.strip() or uploaded_image):
        if auto_submit:
            st.session_state.pending_user_input = ""
        st.session_state.qcount += 1
        lang_instruction = LANGUAGES[st.session_state.selected_language]["lang_instruction"]

        if effective_user_input.strip():
            conv_text = " ".join([m.get("content", "") for m in st.session_state.messages if m.get("type") == "text"])
            if is_meta_text(effective_user_input):
                # Pasted docs / test plans / app-feedback — not a symptom report.
                st.session_state.emergency_detected = False
                st.session_state.emergency_reason = ""
                st.session_state.triage_assessment = None
            else:
                is_emerg, reason = detect_emergency(effective_user_input, conv_text)
                if is_emerg:
                    st.session_state.emergency_detected = True
                    st.session_state.emergency_reason = reason
                st.session_state.triage_assessment = assess_triage_tier(effective_user_input, conv_text, st.session_state.patient_memory)

        if uploaded_image:
            is_pdf = uploaded_image.name.lower().endswith(".pdf")
            if is_pdf:
                st.session_state.messages.append({"role": "user", "type": "pdf", "content": (effective_user_input.strip() + " " if effective_user_input.strip() else "") + "[PDF: " + uploaded_image.name + "]", "ts": _msg_now_ts()})
                with st.spinner("Reading your medical report..."):
                    pdf_text = extract_pdf_text(uploaded_image)
                    if not pdf_text:
                        reply = "I had trouble reading that PDF. It might be image-based (scanned) rather than text-based. Could you try uploading it as a JPEG or PNG image instead?"
                        engine_used = "system"
                    else:
                        st.session_state.last_pdf_context = pdf_text
                        st.session_state.last_pdf_name = uploaded_image.name
                        reply, engine_used = medichat_pdf_analysis(effective_user_input, pdf_text, st.session_state.messages, lang_instruction)
                        reply = strip_excessive_disclaimers(reply)
                st.session_state.last_sources = ["PDF Report Analysis"]
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply, "sources": st.session_state.last_sources, "confidence": "medium", "confidence_pct": 75, "engine": engine_used, "ts": _msg_now_ts()})
                st.session_state.uploader_key += 1
                st.session_state.home_show_vision_upload = False
            else:
                st.session_state.messages.append({"role": "user", "type": "image", "content": effective_user_input.strip(), "ts": _msg_now_ts()})
                if looks_like_prescription_request(effective_user_input):
                    with st.spinner("Reading prescription handwriting..."):
                        uploaded_image.seek(0)
                        rx_result = read_prescription(
                            uploaded_image.read(),
                            user_note=effective_user_input,
                            lang_instruction=lang_instruction,
                        )
                        reply = strip_excessive_disclaimers(rx_result.get("reading", ""))
                        vision_engine = "rx-reader:" + rx_result.get("model_used", "unknown")
                        conf_map = {"high": 88, "medium": 72, "low": 56}
                        conf_level = rx_result.get("overall_confidence", "medium")
                        conf_pct = conf_map.get(conf_level, 70)
                    st.session_state.last_image_context = (
                        "User uploaded prescription image: " + uploaded_image.name + "\n"
                        "User's prompt: " + (effective_user_input.strip() if effective_user_input.strip() else "(no question provided)") + "\n"
                        "Prescription transcription: " + reply
                    )
                    st.session_state.last_sources = ["Prescription Reader"]
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply, "sources": st.session_state.last_sources, "confidence": conf_level, "confidence_pct": conf_pct, "engine": vision_engine, "ts": _msg_now_ts()})
                else:
                    with st.spinner("Analysing your image..."):
                        uploaded_image.seek(0)
                        reply, vision_engine = medichat_vision(effective_user_input, encode_image(uploaded_image), st.session_state.messages, lang_instruction)
                        reply = strip_excessive_disclaimers(reply)
                    st.session_state.last_image_context = (
                        "User uploaded image: " + uploaded_image.name + "\n"
                        "User's question about image: " + (effective_user_input.strip() if effective_user_input.strip() else "(no question, just the image)") + "\n"
                        "Your visual analysis: " + reply
                    )
                    st.session_state.last_sources = ["Image Analysis"]
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply, "sources": st.session_state.last_sources, "confidence": "medium", "confidence_pct": 75, "engine": vision_engine, "ts": _msg_now_ts()})
                st.session_state.uploader_key += 1
                st.session_state.home_show_vision_upload = False
        else:
            user_msg = {"role": "user", "type": "text", "content": effective_user_input.strip()}
            st.session_state.messages.append(user_msg)
            _t0 = time.time()

            name_for_rag = "" if st.session_state.patient_name == "Guest" else st.session_state.patient_name

            # Build cross-chat summary so the AI knows about the patient's other conversations.
            past_chats_summary = ""
            if st.session_state.is_authenticated and st.session_state.user_email_hash:
                _all_convs = list_conversations(st.session_state.user_email_hash, limit=10)
                _other = [c for c in _all_convs if c["id"] != st.session_state.current_conversation_id]
                if _other:
                    _lines = []
                    for c in _other[:5]:
                        _topic = (c.get("first_user_msg") or c.get("title") or "").strip().replace("\n", " ")
                        _outcome = (c.get("last_assistant_msg") or "").strip().replace("\n", " ")
                        if not _topic:
                            continue
                        _entry = "- Topic: " + _topic[:200]
                        if _outcome:
                            _entry += "\n  Last response summary: " + _outcome[:240]
                        _lines.append(_entry)
                    past_chats_summary = "\n".join(_lines)

            # Ensure _bot_avatar_html is always defined regardless of which
            # history-rendering branch executed above (the elif/pass branch
            # skips the definition when messages were empty at render time).
            if "_bot_avatar_html" not in dir():
                _bot_avatar_uri_fallback = get_brand_logo_data_uri()
                _bot_avatar_html = (
                    '<div class="av av-bot av-bot-image"><img src="' + _bot_avatar_uri_fallback + '" alt="MediChat AI"></div>'
                    if _bot_avatar_uri_fallback
                    else '<div class="av av-bot">M</div>'
                )

            with st.spinner("MediChat is analysing"):
                stream_placeholder = st.empty()
                final_text = ""
                stream_metadata = None
            
                try:
                    for event in medichat_rag_stream(effective_user_input, st.session_state.messages, lang_instruction, name_for_rag, st.session_state.get("last_pdf_context", ""), st.session_state.get("last_image_context", ""), past_chats_summary):
                        kind = event[0]
                        if kind == "chunk":
                            final_text = event[2]
                            stream_placeholder.markdown(
                                '<div class="bot-wrap">' + _bot_avatar_html + '<div class="bot-stack">'
                                '<div class="bot-bubble">'
                                '<div class="bot-bubble-head">'
                                '<span class="bot-bubble-name">MediChat AI</span>'
                                '<span class="bot-bubble-spark material-symbols-rounded">auto_awesome</span>'
                                '</div>' + markdown_to_html(final_text) + '<span class="stream-cursor"></span></div>'
                                '</div>'
                                '</div>', unsafe_allow_html=True
                            )
                        elif kind == "done":
                            final_text = event[1]
                            stream_metadata = event[2]
                
                    stream_placeholder.empty()
                
                except Exception as streaming_fault:
                    import traceback as _tb
                    _tb.print_exc()
                    logger.warning("Streaming crash detail: %s: %s", type(streaming_fault).__name__, streaming_fault)
                    stream_placeholder.empty()
                    _err_msg = str(streaming_fault) or "Unknown internal error"
                    st.markdown(
                        f'<div class="md-mini-error">MediChat had trouble generating a response. '
                        f'Error: {_err_msg[:300]}. Please try again or check your API keys in Streamlit secrets.</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()

            final_text = strip_excessive_disclaimers(final_text)

            if stream_metadata is None:
                stream_metadata = {"memory": st.session_state.patient_memory, "sources": [], "confidence": "low", "confidence_pct": 0, "engine": "unknown"}

            memory = stream_metadata["memory"]
            sources = stream_metadata["sources"]
            conf_level = stream_metadata["confidence"]
            conf_pct = stream_metadata["confidence_pct"]
            engine_used = stream_metadata.get("engine", "unknown")
            # Merge fresh extraction with existing profile memory so facts
            # from past chats persist into new ones.
            existing_mem = st.session_state.patient_memory or {}
            merged_mem = {}
            for _key in ("symptoms", "conditions", "medications"):
                _combined = list(dict.fromkeys((existing_mem.get(_key, []) or []) + (memory.get(_key, []) or [])))
                merged_mem[_key] = _combined[:30]
            st.session_state.patient_memory = merged_mem
            memory = merged_mem
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
                    alert_block += "\n- **" + a["drug"] + "**, given your " + ", ".join(a["conditions"]) + ": " + a["warning"]
                final_text = final_text + alert_block

            _log_entry = {
                "query": effective_user_input.strip(),
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

            st.session_state.messages.append({"role": "assistant", "type": "text", "content": final_text, "sources": sources, "confidence": conf_level, "confidence_pct": conf_pct, "engine": engine_used, "ts": _msg_now_ts()})

        if st.session_state.is_authenticated and st.session_state.user_email_hash:
            persist_profile_state(
                st.session_state.user_email_hash,
                patient_memory=st.session_state.patient_memory,
                name=st.session_state.patient_name or None,
                language=st.session_state.selected_language,
            )
            new_id = save_conversation(
                st.session_state.user_email_hash,
                st.session_state.current_conversation_id or "",
                st.session_state.messages,
            )
            if new_id:
                st.session_state.current_conversation_id = new_id
        st.session_state.chat_input_key = st.session_state.get("chat_input_key", 0) + 1
        st.rerun()

def render_eval():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1f2937,#111827);color:white;padding:0.7rem 1.2rem;border-radius:12px;margin-bottom:1rem;display:flex;align-items:center;justify-content:space-between;">'
        '<div style="display:flex;align-items:center;gap:0.5rem;"><span style="font-size:1rem;">🔒</span><span style="font-weight:600;font-size:0.9rem;">Admin Mode. Research & Evaluation Dashboard</span></div>'
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
        st.info("Live data from Firestore, aggregated from all MediChat patients (anonymised). Total records: " + str(len(logs)))
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
                '<div style="color:#334155;margin-bottom:0.2rem;font-weight:600;">Query #' + str(total_queries - i + 1) + ', ' + str(query_len) + ' words</div>'
                '<div style="display:flex;gap:0.8rem;font-size:0.7rem;color:#64748b;flex-wrap:wrap;">'
                '<span style="color:' + conf_color + ';font-weight:600;">● ' + str(l["confidence_pct"]) + '% conf</span>'
                '<span>⏱ ' + str(l["response_time"]) + 's</span>'
                '<span>📚 ' + ", ".join(l.get("sources", []) or ["-"]) + '</span>'
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
                        ", ".join(l.get("sources", [])) or "-",
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

def render_history():
    # ── Full chat history list ──────────────────────────────────────
    st.markdown('<div class="md-greet-wrap"><div class="md-greet">Your Chats</div>'
                '<div class="md-subgreet">Every conversation you have had with MediChat. Open one to continue, or start a new chat.</div></div>',
                unsafe_allow_html=True)
    if st.button("← Back to home", key="hist_back"):
        st.session_state.mode = "chat"
        st.rerun()
    if st.session_state.is_authenticated and st.session_state.user_email_hash:
        _hist = list_conversations(st.session_state.user_email_hash, limit=200)
        if not _hist:
            st.markdown('<div class="md-rcard" style="text-align:center;color:var(--md-text-3);font-style:italic;padding:1.6rem;">No past chats yet. Start one from the home screen.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="md-history-list">', unsafe_allow_html=True)
            for _h in _hist:
                _ht = (_h.get("title") or "Chat")[:80]
                _hc = _h.get("message_count", 0)
                _hu = _h.get("last_updated")
                try:
                    if _hu and hasattr(_hu, "strftime"):
                        _ago_disp = _hu.strftime("%d %b %Y · %H:%M")
                    else:
                        _ago_disp = ""
                except Exception:
                    _ago_disp = ""
                _preview = (_h.get("first_user_msg") or "")[:120]
                st.markdown(
                    '<div class="md-history-row">'
                    '<div class="md-history-bubble">💬</div>'
                    '<div class="md-history-mid">'
                    '<div class="md-history-title">' + ui_text(_ht, 80) + '</div>'
                    '<div class="md-history-meta">' + str(_hc) + ' messages · ' + ui_text(_ago_disp, 40) + '</div>'
                    + ('<div class="md-history-preview">' + ui_text(_preview, 140) + '</div>' if _preview else '') +
                    '</div></div>',
                    unsafe_allow_html=True
                )
                hc1, hc2 = st.columns([4, 1])
                with hc1:
                    if st.button("Open", key="hist_open_" + _h["id"], use_container_width=True):
                        conv = load_conversation(st.session_state.user_email_hash, _h["id"])
                        if conv is not None:
                            st.session_state.current_conversation_id = _h["id"]
                            st.session_state.messages = conv.get("messages", []) or []
                            st.session_state.qcount = sum(1 for m in st.session_state.messages if m.get("role") == "user")
                            st.session_state.feedback = {}
                            st.session_state.last_sources = []
                            st.session_state.emergency_detected = False
                            st.session_state.mode = "chat"
                            st.rerun()
                with hc2:
                    if st.button("Delete", key="hist_del_" + _h["id"], use_container_width=True):
                        delete_conversation(st.session_state.user_email_hash, _h["id"])
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="md-rcard" style="text-align:center;color:var(--md-text-3);font-style:italic;padding:1.6rem;">Sign in to keep and review your chat history.</div>', unsafe_allow_html=True)

def render_overview():
    # ── Health Overview ─────────────────────────────────────────────
    st.markdown(
        '<div class="md-page-hero md-page-hero-overview">'
        '<div class="md-page-hero-ic"><span class="material-symbols-rounded">monitoring</span></div>'
        '<div class="md-page-hero-text">'
        '<div class="md-page-hero-title">Health Overview</div>'
        '<div class="md-page-hero-sub">A live snapshot of what we know about you and what you have logged today.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    today = get_daily_metrics()
    water_n = int(today.get("water_glasses", 0) or 0)
    sleep_h = today.get("sleep_hours")
    steps_n = today.get("steps")
    hr_n = today.get("heart_rate_resting")

    # Wearable sync placeholder (no simulated data)
    st.markdown(
        '<div class="md-wearable-card">'
        '<div class="md-wearable-icon">⌚</div>'
        '<div class="md-wearable-body">'
        '<div class="md-wearable-title">Wearable data source: Not connected</div>'
        '<div class="md-wearable-desc">Connect a wearable to track heart rate, steps, and activity automatically. Until then, health overview only uses your manually logged data.</div>'
        '<div class="md-wearable-actions">'
        '<span class="md-wearable-pill">Bluetooth</span>'
        '<span class="md-wearable-pill">Apple Health</span>'
        '<span class="md-wearable-pill">Google Fit</span>'
        '<span class="md-wearable-pill md-wearable-soon">Not connected</span>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    ov_c, ov_d = st.columns(2)
    with ov_c:
        sleep_disp = (str(sleep_h) + " hrs") if sleep_h else "-"
        st.markdown(
            '<div class="md-rcard"><div class="md-metric-row" style="border:none;">'
            '<div class="md-metric-icon md-hp-violet">🌙</div>'
            '<div class="md-metric-mid"><div class="md-metric-label">Sleep last night</div>'
            '<div class="md-metric-value">' + ui_text(sleep_disp, 20) + '</div></div></div></div>',
            unsafe_allow_html=True
        )
        with st.form("sleep_form_today", clear_on_submit=True):
            sl_in = st.number_input("Log sleep (hours)", min_value=0.0, max_value=16.0, step=0.5, value=float(sleep_h) if sleep_h else 7.0, key="sleep_input")
            if st.form_submit_button("Save sleep", use_container_width=True):
                update_daily_metric("sleep_hours", float(sl_in))
                st.rerun()
    with ov_d:
        pct_water = int(round((water_n / 8) * 100)) if water_n else 0
        st.markdown(
            '<div class="md-rcard"><div class="md-metric-row" style="border:none;">'
            '<div class="md-metric-icon md-hp-blue">💧</div>'
            '<div class="md-metric-mid"><div class="md-metric-label">Water today</div>'
            '<div class="md-metric-value">' + str(water_n) + ' / 8 glasses</div></div>'
            '<div class="md-metric-status md-status-info">' + str(pct_water) + '%</div></div></div>',
            unsafe_allow_html=True
        )
        wcol_a, wcol_b = st.columns(2)
        with wcol_a:
            if st.button("+1 glass", key="water_inc", use_container_width=True):
                update_daily_metric("water_glasses", water_n + 1)
                st.rerun()
        with wcol_b:
            if st.button("Reset", key="water_reset", use_container_width=True):
                update_daily_metric("water_glasses", 0)
                st.rerun()

    ov_e, ov_f = st.columns(2)
    with ov_e:
        steps_disp = (f"{int(steps_n):,} / 10,000") if steps_n else "-"
        steps_pct = min(100, int(round((int(steps_n) / 10000) * 100))) if steps_n else 0
        st.markdown(
            '<div class="md-rcard"><div class="md-metric-row" style="border:none;">'
            '<div class="md-metric-icon md-hp-green">🚶</div>'
            '<div class="md-metric-mid"><div class="md-metric-label">Steps today</div>'
            '<div class="md-metric-value">' + ui_text(steps_disp, 24) + '</div></div>'
            + ('<div class="md-metric-status md-status-info">' + str(steps_pct) + '%</div>' if steps_n else '')
            + '</div></div>',
            unsafe_allow_html=True
        )
        with st.form("steps_form_today", clear_on_submit=True):
            st_in = st.number_input("Log steps today", min_value=0, max_value=100000, step=500, value=int(steps_n) if steps_n else 0, key="steps_input")
            if st.form_submit_button("Save steps", use_container_width=True):
                update_daily_metric("steps", int(st_in))
                st.rerun()
    with ov_f:
        hr_disp = (str(int(hr_n)) + " BPM") if hr_n else "-"
        _hr_lbl, _hr_cls_p = heart_rate_status(hr_n)
        hr_status_html = ('<div class="md-metric-status ' + (_hr_cls_p or "md-status-info") + '">' + (_hr_lbl or "") + '</div>') if hr_n else ''
        st.markdown(
            '<div class="md-rcard"><div class="md-metric-row" style="border:none;">'
            '<div class="md-metric-icon md-hp-pink">❤</div>'
            '<div class="md-metric-mid"><div class="md-metric-label">Resting heart rate</div>'
            '<div class="md-metric-value">' + ui_text(hr_disp, 16) + '</div></div>'
            + hr_status_html
            + '</div></div>',
            unsafe_allow_html=True
        )
        with st.form("hr_form_today", clear_on_submit=True):
            hr_in = st.number_input("Log resting HR (BPM)", min_value=30, max_value=220, step=1, value=int(hr_n) if hr_n else 70, key="hr_input")
            if st.form_submit_button("Save heart rate", use_container_width=True):
                update_daily_metric("heart_rate_resting", int(hr_in))
                st.rerun()

    # 7-day history table
    st.markdown('<div class="md-smart-head" style="margin-top:1rem;"><div class="md-smart-title">Last 7 days</div></div>', unsafe_allow_html=True)
    history = get_metrics_history(7)
    rows_html = '<div class="md-rcard"><div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.4rem;font-size:0.78rem;">'
    rows_html += '<div style="font-weight:700;color:var(--md-text-2);">Day</div><div style="font-weight:700;color:var(--md-text-2);">Water</div><div style="font-weight:700;color:var(--md-text-2);">Sleep</div>'
    for d, m in history:
        sl = m.get("sleep_hours")
        rows_html += '<div>' + ui_text(d, 20) + '</div>'
        rows_html += '<div>' + str(int(m.get("water_glasses", 0) or 0)) + ' glasses</div>'
        rows_html += '<div>' + ui_text((str(sl) + " h" if sl else "-"), 20) + '</div>'
    rows_html += '</div></div>'
    st.markdown(rows_html, unsafe_allow_html=True)

def render_medications():
    # ── Medications ─────────────────────────────────────────────────
    st.markdown(
        '<div class="md-page-hero md-page-hero-meds">'
        '<div class="md-page-hero-ic"><span class="material-symbols-rounded">pill</span></div>'
        '<div class="md-page-hero-text">'
        '<div class="md-page-hero-title">Medications</div>'
        '<div class="md-page-hero-sub">Keep track of what you take and when. Stored to your profile so MediChat can reference it during chats.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    with st.form("add_med_form", clear_on_submit=True):
        st.markdown(
            '<div class="md-form-intro">Add medication</div>'
            '<div class="md-form-sub">Capture dose, timing and notes so MediChat can reference them in future chats.</div>',
            unsafe_allow_html=True
        )
        mc1, mc2 = st.columns(2)
        with mc1:
            m_name = st.text_input("Name", placeholder="e.g. Metformin")
            m_freq = st.selectbox("Frequency", ["Once daily", "Twice daily", "Three times daily", "Four times daily", "As needed", "Weekly"])
        with mc2:
            m_dose = st.text_input("Dose", placeholder="e.g. 500 mg")
            m_time = st.text_input("Time(s) of day", placeholder="e.g. Morning, 8 pm")
        m_notes = st.text_area("Notes (optional)", placeholder="Take with food, etc.", height=80)
        if st.form_submit_button("Save medication", use_container_width=True, type="primary", icon=":material/save:"):
            if add_medication(m_name, m_dose, m_freq, m_time, m_notes):
                st.success("Medication added.")
                st.rerun()
            else:
                st.error("Please enter a name.")

    meds = list_medications()
    if not meds:
        st.markdown('<div class="md-rcard" style="text-align:center;color:var(--md-text-3);font-style:italic;padding:1.6rem;">No medications added yet.</div>', unsafe_allow_html=True)
    else:
        for m in meds:
            mh = '<div class="md-rcard"><div style="display:flex;align-items:flex-start;gap:0.8rem;">'
            mh += '<div class="md-metric-icon md-hp-violet" style="width:42px;height:42px;flex-shrink:0;">💊</div>'
            mh += '<div style="flex:1;min-width:0;">'
            mh += '<div style="font-weight:700;color:var(--md-text-1);font-size:0.98rem;">' + ui_text(m.get("name", ""), 90) + '</div>'
            mh += '<div style="font-size:0.78rem;color:var(--md-text-2);margin-top:0.15rem;">' + ui_text(m.get("dose", "") or "Dose not specified", 40) + ' · ' + ui_text(m.get("frequency", ""), 40) + (" · " + ui_text(m.get("time_of_day"), 40) if m.get("time_of_day") else "") + '</div>'
            if m.get("notes"):
                mh += '<div style="font-size:0.76rem;color:var(--md-text-2);margin-top:0.3rem;font-style:italic;">' + ui_text(m.get("notes"), 260) + '</div>'
            mh += '</div></div></div>'
            st.markdown(mh, unsafe_allow_html=True)
            if st.button("Remove", key="del_med_" + str(m.get("id", "")), use_container_width=False):
                delete_medication(m.get("id"))
                st.rerun()

def render_appointments():
    # ── Appointments ────────────────────────────────────────────────
    st.markdown(
        '<div class="md-page-hero md-page-hero-appts">'
        '<div class="md-page-hero-ic"><span class="material-symbols-rounded">calendar_month</span></div>'
        '<div class="md-page-hero-text">'
        '<div class="md-page-hero-title">Appointments</div>'
        '<div class="md-page-hero-sub">Upcoming visits and reminders. Stored to your profile so MediChat can reference them in chats.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    with st.form("add_appt_form", clear_on_submit=True):
        st.markdown(
            '<div class="md-form-intro">Add appointment</div>'
            '<div class="md-form-sub">Save upcoming visits, reminders and notes in one place.</div>',
            unsafe_allow_html=True
        )
        ac1, ac2 = st.columns(2)
        with ac1:
            a_title = st.text_input("Title", placeholder="e.g. GP follow-up")
            a_date = st.date_input("Date", min_value=_date.today())
            a_time = st.time_input("Time", value=datetime.now().time().replace(minute=0, second=0, microsecond=0))
        with ac2:
            a_doc = st.text_input("Doctor / clinician", placeholder="e.g. Dr Patel")
            a_loc = st.text_input("Location", placeholder="e.g. Melbourne Health Centre")
        a_notes = st.text_area("Notes (optional)", placeholder="Bring previous lab results, etc.", height=80)
        if st.form_submit_button("Save appointment", use_container_width=True, type="primary", icon=":material/save:"):
            iso = datetime.combine(a_date, a_time).isoformat(timespec="minutes")
            if add_appointment(a_title, iso, a_doc, a_loc, a_notes):
                st.success("Appointment added.")
                st.rerun()
            else:
                st.error("Please enter a title and date.")

    appts = list_appointments()
    if not appts:
        st.markdown('<div class="md-rcard" style="text-align:center;color:var(--md-text-3);font-style:italic;padding:1.6rem;">No appointments scheduled yet.</div>', unsafe_allow_html=True)
    else:
        # Sort by date ascending
        try:
            appts_sorted = sorted(appts, key=lambda a: a.get("date", ""))
        except Exception:
            appts_sorted = appts
        now_iso = datetime.now().isoformat(timespec="minutes")
        for a in appts_sorted:
            past = a.get("date", "") < now_iso
            status_cls = "md-status-warn" if past else "md-status-info"
            status_lbl = "Past" if past else "Upcoming"
            try:
                _dt = datetime.fromisoformat(a.get("date"))
                date_disp = _dt.strftime("%a %d %b %Y · %I:%M %p")
            except Exception:
                date_disp = a.get("date", "")
            ah = '<div class="md-rcard"><div style="display:flex;align-items:flex-start;gap:0.8rem;">'
            ah += '<div class="md-metric-icon md-hp-blue" style="flex-shrink:0;">📅</div>'
            ah += '<div style="flex:1;min-width:0;">'
            ah += '<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;"><div style="font-weight:700;color:var(--md-text-1);font-size:0.98rem;">' + ui_text(a.get("title", ""), 90) + '</div>'
            ah += '<span class="md-metric-status ' + status_cls + '">' + status_lbl + '</span></div>'
            ah += '<div style="font-size:0.78rem;color:var(--md-text-2);margin-top:0.15rem;">' + ui_text(date_disp, 60) + '</div>'
            details = []
            if a.get("doctor"):
                details.append("👨‍⚕️ " + ui_text(a["doctor"], 60))
            if a.get("location"):
                details.append("📍 " + ui_text(a["location"], 80))
            if details:
                ah += '<div style="font-size:0.78rem;color:var(--md-text-2);margin-top:0.2rem;">' + " &nbsp; ".join(details) + '</div>'
            if a.get("notes"):
                ah += '<div style="font-size:0.76rem;color:var(--md-text-2);margin-top:0.3rem;font-style:italic;">' + ui_text(a.get("notes"), 260) + '</div>'
            ah += '</div></div></div>'
            st.markdown(ah, unsafe_allow_html=True)
            if st.button("Remove", key="del_appt_" + str(a.get("id", "")), use_container_width=False):
                delete_appointment(a.get("id"))
                st.rerun()

def render_records():
    # ── Health Records ──────────────────────────────────────────────
    st.markdown(
        '<div class="md-page-hero md-page-hero-records">'
        '<div class="md-page-hero-ic"><span class="material-symbols-rounded">medical_information</span></div>'
        '<div class="md-page-hero-text">'
        '<div class="md-page-hero-title">Health Records</div>'
        '<div class="md-page-hero-sub">Upload medical PDFs or images. We extract text and store metadata only. File contents stay on your device unless you choose to keep an Ai summary.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    with st.form("rec_upload_form", clear_on_submit=True):
        rec_file = st.file_uploader("Drop a medical record here", type=["pdf", "jpg", "jpeg", "png"], key="hr_upload")
        rec_label = st.text_input("Label this record", placeholder="e.g. Blood test - April 2026")
        rec_keep_summary = st.checkbox("Generate and save an Ai summary (recommended)", value=True)
        if st.form_submit_button("Save record", use_container_width=True, type="primary", icon=":material/save:"):
            if not rec_file:
                st.error("Please pick a file.")
            elif not rec_label.strip():
                st.error("Please add a label.")
            else:
                size = len(rec_file.getbuffer())
                summary = ""
                if rec_keep_summary:
                    try:
                        if rec_file.name.lower().endswith(".pdf"):
                            txt = extract_pdf_text(rec_file)
                            if txt:
                                ai_resp, _ = medichat_pdf_analysis("Briefly summarise the key findings of this report in plain language for a patient.", txt[:6000], st.session_state.messages)
                                summary = strip_excessive_disclaimers(ai_resp or "")[:1400]
                    except Exception as _e:
                        logger.warning("record summary failed: %s", _e)
                if add_health_record(rec_label, rec_file.type or rec_file.name.split(".")[-1].upper(), size, summary):
                    st.success("Record saved.")
                    st.rerun()

    records = list_health_records()
    if not records:
        st.markdown('<div class="md-rcard" style="text-align:center;color:var(--md-text-3);font-style:italic;padding:1.6rem;">No records uploaded yet.</div>', unsafe_allow_html=True)
    else:
        records_sorted = sorted(records, key=lambda r: r.get("uploaded_at", ""), reverse=True)
        for r in records_sorted:
            ic = "📄" if "pdf" in (r.get("file_type", "").lower()) else "🖼"
            try:
                kb = round(r.get("size_bytes", 0) / 1024, 1)
                size_lbl = (str(kb) + " KB") if kb < 1024 else (str(round(kb / 1024, 1)) + " MB")
            except Exception:
                size_lbl = ""
            uploaded = r.get("uploaded_at", "")[:16].replace("T", " ")
            rh = '<div class="md-rcard"><div style="display:flex;align-items:flex-start;gap:0.8rem;">'
            rh += '<div class="md-metric-icon md-hp-green" style="width:42px;height:42px;flex-shrink:0;font-size:1.2rem;">' + ic + '</div>'
            rh += '<div style="flex:1;min-width:0;">'
            rh += '<div style="font-weight:700;color:var(--md-text-1);font-size:0.96rem;">' + ui_text(r.get("name", ""), 120) + '</div>'
            rh += '<div style="font-size:0.74rem;color:var(--md-text-3);margin-top:0.15rem;">' + ui_text(r.get("file_type", ""), 30) + ' · ' + ui_text(size_lbl, 20) + ' · uploaded ' + ui_text(uploaded, 30) + '</div>'
            if r.get("summary"):
                rh += '<details style="margin-top:0.4rem;"><summary style="cursor:pointer;font-size:0.78rem;color:var(--md-brand-2);font-weight:600;">View Ai summary</summary><div style="font-size:0.8rem;color:var(--md-text-2);margin-top:0.4rem;line-height:1.5;">' + ui_lines(r.get("summary")) + '</div></details>'
            rh += '</div></div></div>'
            st.markdown(rh, unsafe_allow_html=True)
            if st.button("Remove", key="del_rec_" + str(r.get("id", "")), use_container_width=False):
                delete_health_record(r.get("id"))
                st.rerun()

def render_rx_reader():
    # ── Prescription Reader ────────────────────────────────────────
    rx_uploader_widget_key = "rx_uploader_" + str(st.session_state.get("rx_uploader_key", 0))
    st.markdown(
        '<div class="md-page-hero md-page-hero-rx">'
        '<div class="md-page-hero-ic md-page-hero-ic-rx"><span class="md-rx-glyph">&#8478;</span></div>'
        '<div class="md-page-hero-text">'
        '<div class="md-page-hero-title">Prescription Reader</div>'
        '<div class="md-page-hero-sub">Upload a handwritten prescription photo. MediChat will transcribe text only, with confidence grading and an AU drug-name cross-check.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    with st.form("rx_reader_form", clear_on_submit=False):
        rx_img = st.file_uploader("Drop your prescription image here", type=["jpg", "jpeg", "png"], key=rx_uploader_widget_key)
        rx_note = st.text_input("Optional context", placeholder="e.g. GP script from today, hard to read medication line")
        rx_btn_a, rx_btn_b = st.columns([3, 1])
        with rx_btn_a:
            rx_submit = st.form_submit_button("Read prescription", use_container_width=True, type="primary", icon=":material/arrow_forward:")
        with rx_btn_b:
            rx_clear = st.form_submit_button("Clear", use_container_width=True)

    if rx_clear:
        reset_prescription_reader_state()

    if rx_submit:
        if not rx_img:
            st.error("Please upload a prescription image first.")
        else:
            with st.spinner("Reading prescription text..."):
                try:
                    rx_img.seek(0)
                    st.session_state.rx_reader_result = read_prescription(
                        rx_img.read(),
                        user_note=rx_note,
                        lang_instruction=LANGUAGES[st.session_state.selected_language]["lang_instruction"],
                    )
                except Exception as e:
                    st.session_state.rx_reader_result = None
                    st.error("Prescription reader failed. Please retry with a clearer image. Error: " + str(e))

    rx_result = st.session_state.get("rx_reader_result")
    if rx_result:
        conf = (rx_result.get("overall_confidence") or "unknown").upper()
        model_used = rx_result.get("model_used", "unknown")
        st.markdown(
            '<div class="md-rcard" style="margin-top:0.55rem;">'
            '<div style="display:flex;gap:0.6rem;flex-wrap:wrap;align-items:center;">'
            '<span class="md-metric-status md-status-info">Model: ' + ui_text(model_used, 40) + '</span>'
            '<span class="md-metric-status md-status-good">Overall confidence: ' + ui_text(conf, 10) + '</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="md-rcard" style="margin-top:0.65rem;">' + markdown_to_html(rx_result.get("reading", "")) + '</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="md-inline-note" style="margin-top:0.65rem;">'
            'Safety guardrail: this feature is transcription only. It does not prescribe therapy or confirm diagnosis. '
            'Always confirm the script with a pharmacist or the prescriber.'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("Upload a new prescription", key="rx_reader_reset_after_result", use_container_width=True):
            reset_prescription_reader_state()

def render_privacy():
    # ── Privacy & Consent ─────────────────────────────────────────
    st.markdown(
        '<div class="md-greet-wrap"><div class="md-greet">Privacy & Consent</div>'
        '<div class="md-subgreet">MediChat uses HIPAA-style technical safeguards and is designed to align with the Australian Privacy Principles (APPs) and NDB obligations.</div></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="md-rcard">'
        '<div style="font-weight:800;color:var(--md-text-1);font-size:0.95rem;margin-bottom:0.45rem;">Alignment with Data Handling Standards</div>'
        '<div style="font-size:0.84rem;color:var(--md-text-2);line-height:1.6;">'
        '<strong>APP-first design:</strong> collection minimisation, purpose limitation, patient access and correction workflows, and secure retention controls.<br>'
        '<strong>NDB-ready response:</strong> breach assessment and notification process aligned with the Australian Notifiable Data Breaches scheme for eligible incidents.<br>'
        '<strong>HIPAA-style safeguards:</strong> role-aware access controls, auditability, and encrypted transit/storage patterns where supported by deployment infrastructure.<br>'
        '<strong>Consent & transparency:</strong> users can continue in Guest mode with lower data retention, or create a profile with explicit consent flow.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="md-rcard" style="margin-top:0.7rem;">'
        '<div style="font-weight:800;color:var(--md-text-1);font-size:0.95rem;margin-bottom:0.45rem;">Clinical Guardrails</div>'
        '<div style="font-size:0.84rem;color:var(--md-text-2);line-height:1.6;">'
        'MediChat is intentionally restricted from issuing final diagnoses or exact medication dose instructions. '
        'It provides educational guidance, triage cues, and clear escalation advice to licensed clinicians.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="md-rcard" style="margin-top:0.7rem;">'
        '<div style="font-size:0.84rem;color:var(--md-text-2);line-height:1.6;">'
        'Privacy policy link: <a href="' + ui_escape(PRIVACY_POLICY_URL) + '" target="_blank">Open Privacy Policy (APP + NDB)</a>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

def render_insights():
    # ── Ai Insights ─────────────────────────────────────────────────
    st.markdown(
        '<div class="md-page-hero md-page-hero-insights">'
        '<div class="md-page-hero-ic"><span class="material-symbols-rounded">auto_awesome</span></div>'
        '<div class="md-page-hero-text">'
        '<div class="md-page-hero-title">Ai Insights</div>'
        '<div class="md-page-hero-sub">Personalised observations generated from what you have logged. Refreshes each time you visit.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    insights = []
    mem = st.session_state.patient_memory or {}
    meds = list_medications()
    appts = list_appointments()
    records = list_health_records()
    saved_conversations = []
    if st.session_state.is_authenticated and st.session_state.user_email_hash:
        saved_conversations = list_conversations(st.session_state.user_email_hash, limit=20)
    history = get_metrics_history(7)
    today_m = get_daily_metrics()

    # Hydration
    water_today = int(today_m.get("water_glasses", 0) or 0)
    sleeps = [m.get("sleep_hours") for _, m in history if m.get("sleep_hours") is not None]
    has_profile_data = bool(mem.get("symptoms") or mem.get("conditions") or mem.get("medications"))
    has_logged_metrics = (water_today > 0 or len(sleeps) > 0)
    has_records = bool(records)
    has_chat_history = bool(saved_conversations)

    if not (has_profile_data or has_logged_metrics or has_records or has_chat_history or meds or appts):
        st.markdown(
            '<div class="md-empty-card">'
            '<div class="md-empty-icon"><span class="material-symbols-rounded">lightbulb</span></div>'
            '<div class="md-empty-title">No insights yet</div>'
            '<div class="md-empty-copy">Log a few health details such as water, sleep, steps, medications, or a record, and MediChat will start generating personalised observations here.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.stop()

    if water_today < 4:
        insights.append(("💧", "Hydration is low today",
            "You have logged " + str(water_today) + " glasses today. Aim for at least 6-8 glasses across the day, more if active.",
            "warn"))
    elif water_today >= 6:
        insights.append(("💧", "Great hydration today",
            "You are at " + str(water_today) + " glasses. Keep it steady through the evening.", "good"))

    # Sleep
    if len(sleeps) >= 3:
        avg_sleep = round(sum(sleeps) / len(sleeps), 1)
        if avg_sleep < 6:
            insights.append(("😴", "Sleep is running short",
                "Your last " + str(len(sleeps)) + " logged nights average " + str(avg_sleep) + " hours. Most adults need 7-9.", "warn"))
        elif avg_sleep > 9:
            insights.append(("😴", "Long sleep pattern",
                "Average " + str(avg_sleep) + " hours. Long sleep can sometimes signal infection or low mood, worth mentioning to a GP if it persists.", "info"))
        else:
            insights.append(("😴", "Sleep looks healthy",
                "Averaging " + str(avg_sleep) + " hours over your last " + str(len(sleeps)) + " logged nights.", "good"))

    # Medications & conditions cross-check
    if meds and mem.get("conditions"):
        insights.append(("💊", "Profile is well-populated",
            "You have " + str(len(meds)) + " medication(s) and " + str(len(mem.get("conditions"))) + " condition(s) recorded. MediChat will use these in every chat.", "info"))
    elif meds and not mem.get("conditions"):
        insights.append(("💊", "Add the conditions these medications treat",
            "You have " + str(len(meds)) + " medication(s) recorded but no conditions yet. Telling MediChat why you take each one will improve its advice.", "warn"))
    elif not meds and mem.get("conditions"):
        insights.append(("💊", "Any medications to add?",
            "You have conditions recorded (" + ", ".join(mem.get("conditions", [])[:3]) + ") but no medications yet. Add them so MediChat can flag interactions.", "info"))

    # Upcoming appointment
    now_iso = datetime.now().isoformat(timespec="minutes")
    upcoming = [a for a in appts if a.get("date", "") >= now_iso]
    if upcoming:
        upcoming.sort(key=lambda a: a.get("date", ""))
        nx = upcoming[0]
        try:
            _dt = datetime.fromisoformat(nx.get("date"))
            in_days = (_dt.date() - _date.today()).days
            when = "today" if in_days == 0 else ("tomorrow" if in_days == 1 else "in " + str(in_days) + " days")
        except Exception:
            when = "soon"
        insights.append(("📅", "Upcoming appointment " + when,
            (nx.get("title") or "Appointment") + (" with " + nx.get("doctor", "") if nx.get("doctor") else "") + ". Want a Doctor Visit Summary PDF before then?", "info"))

    # Symptom load
    sym_n = len(mem.get("symptoms", []) or [])
    if sym_n >= 5:
        insights.append(("📌", "Several active symptoms",
            "MediChat has " + str(sym_n) + " symptoms on file from your chats. Consider a Symptoms Checker run to see if a pattern emerges.", "warn"))

    if records:
        insights.append(("📄", "Health records available",
            "You have " + str(len(records)) + " uploaded record(s). MediChat can use these to improve context during chat and summaries.", "info"))

    if saved_conversations:
        insights.append(("💬", "Conversation history linked",
            str(len(saved_conversations)) + " saved conversation(s) are available for continuity of care context.", "info"))

    # AI-generated overall insight (one Claude call) if logged in and has data
    if CLAUDE_ACTIVE and (meds or mem.get("conditions") or len(sleeps) >= 2 or records or saved_conversations):
        try:
            data_lines = []
            if mem.get("conditions"):
                data_lines.append("Conditions: " + ", ".join(mem["conditions"][:6]))
            if meds:
                data_lines.append("Medications: " + ", ".join([m["name"] + " " + m.get("dose", "") for m in meds[:6]]))
            if sleeps:
                data_lines.append("Recent sleep avg: " + str(round(sum(sleeps)/len(sleeps), 1)) + " h over " + str(len(sleeps)) + " nights")
            data_lines.append("Hydration today: " + str(water_today) + "/8 glasses")
            blob = "\n".join(data_lines)
            resp = anthropic_client.messages.create(
                model=CLAUDE_MODEL, max_tokens=200, temperature=0.4,
                system="You are a careful AI health companion. Read the patient's logged data and give ONE clear, kind, evidence-aware insight in 2 short sentences. Avoid alarming language. End with a concrete suggestion. No disclaimer.",
                messages=[{"role": "user", "content": blob}],
            )
            ai_text = (resp.content[0].text or "").strip()
            if ai_text:
                insights.append(("✨", "Personalised observation", ai_text, "info"))
        except Exception as _e:
            logger.warning("AI insights failed: %s", _e)

    if not insights:
        st.markdown('<div class="md-rcard" style="text-align:center;color:var(--md-text-3);padding:1.6rem;">'
                    'Add more health information to generate insights.</div>',
                    unsafe_allow_html=True)
    else:
        for icon, title, body, kind in insights:
            badge_cls = {"good": "md-status-good", "warn": "md-status-warn", "info": "md-status-info"}.get(kind, "md-status-info")
            ih = '<div class="md-rcard"><div style="display:flex;align-items:flex-start;gap:0.8rem;">'
            ih += '<div class="md-metric-icon md-hp-violet" style="width:42px;height:42px;flex-shrink:0;font-size:1.2rem;">' + icon + '</div>'
            ih += '<div style="flex:1;min-width:0;">'
            ih += '<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;"><div style="font-weight:700;color:var(--md-text-1);font-size:0.95rem;">' + ui_text(title, 100) + '</div>'
            ih += '<span class="md-metric-status ' + badge_cls + '">' + ui_text(kind.upper(), 12) + '</span></div>'
            ih += '<div style="font-size:0.85rem;color:var(--md-text-2);margin-top:0.3rem;line-height:1.5;">' + ui_lines(body) + '</div>'
            ih += '</div></div></div>'
            st.markdown(ih, unsafe_allow_html=True)

def render_default():
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
            st.info("**Conditions:** " + data.get("known_conditions", "Not specified"))
            st.info("**Medications:** " + data.get("current_medications", "Not specified"))

        other = data.get("other_symptoms", "")
        if other and other.lower() not in ["no", "none", "n/a", "no other symptoms"]:
            st.info("**Other symptoms:** " + other)
        red_flags = data.get("red_flags", "")
        if red_flags:
            st.info("**Red flags:** " + red_flags)

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

        if parsed.get("safety"):
            st.markdown('<div class="md-inline-note" style="margin-top:0.6rem;">' + ui_text(parsed.get("safety"), 300) + '</div>', unsafe_allow_html=True)

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

        st.markdown(
            '<div class="assessment-card">'
            '<div class="assessment-card-head">'
            '<div class="md-page-hero-ic"><span class="material-symbols-rounded">stethoscope</span></div>'
            '<div class="assessment-head-text">'
            '<div class="assessment-title">' + L["symptom_title"] + '</div>'
            '<div class="assessment-subtitle">' + L["symptom_subtitle"] + '</div>'
            '</div>'
            '</div>'
            '<div class="progress-label"><span>Step ' + str(stage + 1) + ' of ' + str(total) + '</span><span>' + str(progress) + '% complete</span></div>'
            '<div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:' + str(progress) + '%;"></div></div>'
            '</div>',
            unsafe_allow_html=True
        )

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
                            if current["key"] in ("main_symptom", "red_flags"):
                                is_emerg, detected_reason = detect_emergency(opt)
                                if is_emerg:
                                    st.session_state.emergency_detected = True
                                    st.session_state.emergency_reason = detected_reason
                            st.session_state.assessment_stage += 1
                            if st.session_state.assessment_stage >= total:
                                lang_instruction = LANGUAGES[st.session_state.selected_language]["lang_instruction"]
                                with st.spinner("Compiling patient clinical evaluation index..."):
                                    try:
                                        report = generate_assessment_report(st.session_state.assessment_data, lang_instruction)
                                        st.session_state.assessment_report = report
                                        st.session_state.assessment_parsed = parse_report(report)
                                        st.session_state.assessment_complete = True
                                    except Exception as report_exception:
                                        st.error("Report assembly interrupted. Re-verifying parameter states.")
                                        logger.warning("Structural Generation Error: %s", report_exception)
                            st.rerun()

            with st.form(key="assessment_form_" + str(stage), clear_on_submit=True):
                typed = st.text_input("", placeholder="Or type your own answer here...", label_visibility="collapsed")
                ac1, ac2 = st.columns([3, 1])
                with ac1:
                    next_btn = st.form_submit_button(L["next"], use_container_width=True, type="primary", icon=":material/arrow_forward:")
                with ac2:
                    cancel_btn = st.form_submit_button(L["cancel"], use_container_width=True)

            if next_btn and typed.strip():
                # Force localized atomic update to block state race conditions
                st.session_state.assessment_data[current["key"]] = typed.strip()
            
                if current["key"] in ("main_symptom", "red_flags"):
                    is_emerg, detected_reason = detect_emergency(typed)
                    if is_emerg:
                        st.session_state.emergency_detected = True
                        st.session_state.emergency_reason = detected_reason
                    
                st.session_state.assessment_stage += 1
            
                if st.session_state.assessment_stage >= total:
                    lang_instruction = LANGUAGES[st.session_state.selected_language]["lang_instruction"]
                    with st.spinner("Compiling patient clinical evaluation index..."):
                        try:
                            report = generate_assessment_report(st.session_state.assessment_data, lang_instruction)
                            st.session_state.assessment_report = report
                            st.session_state.assessment_parsed = parse_report(report)
                            st.session_state.assessment_complete = True
                        except Exception as report_exception:
                            st.error("Report assembly interrupted. Re-verifying parameter states.")
                            logger.warning("Structural Generation Error: %s", report_exception)
                        
                st.rerun()

            if cancel_btn:
                st.session_state.assessment_stage = 0
                st.session_state.assessment_data = {}
                st.session_state.mode = "chat"
                st.rerun()

_MODE_RENDERERS = {
    "chat": render_chat,
    "eval": render_eval,
    "history": render_history,
    "overview": render_overview,
    "medications": render_medications,
    "appointments": render_appointments,
    "records": render_records,
    "rx_reader": render_rx_reader,
    "privacy": render_privacy,
    "insights": render_insights,
}

_MODE_RENDERERS.get(st.session_state.mode, render_default)()
