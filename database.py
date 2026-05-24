import os
import hashlib
import uuid as _uuid
import json
from datetime import datetime, timedelta, date as _date
import streamlit as st
from clients import CLAUDE_ACTIVE, anthropic_client, CLAUDE_MODEL

# ── Local Fallback DB Configuration ───────────────────────────────────
LOCAL_DB_DIR = os.path.join(os.path.dirname(__file__), "cache")
LOCAL_DB_PATH = os.path.join(LOCAL_DB_DIR, "local_db.json")

def _read_local_db():
    if not os.path.exists(LOCAL_DB_PATH):
        os.makedirs(LOCAL_DB_DIR, exist_ok=True)
        return {"profiles": {}, "conversations": {}, "queries": []}
    try:
        with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Failed to read local DB:", e)
        return {"profiles": {}, "conversations": {}, "queries": []}

def _write_local_db(db):
    try:
        os.makedirs(LOCAL_DB_DIR, exist_ok=True)
        with open(LOCAL_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, default=str)
        return True
    except Exception as e:
        print("Failed to write local DB:", e)
        return False


try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# Firebase for cross-session analytics (optional - fails gracefully if missing)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

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

MEDICAL_REFERENCE_TARGET = max(1000, _safe_int_env("MEDICHAT_REFERENCE_TARGET", 5000))
PRIVACY_POLICY_URL = _safe_secret(
    "PRIVACY_POLICY_URL",
    os.environ.get("PRIVACY_POLICY_URL", "?mode=privacy"),
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
        print("Firebase init failed:", e)
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
        db = _read_local_db()
        return db.get("profiles", {}).get(email_hash)
    try:
        doc = firestore_db.collection("medichat_profiles").document(email_hash).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print("Profile fetch failed:", e)
        return None

def create_profile(email, pin, name=""):
    eh = hash_email(email)
    now_str = datetime.now().isoformat()
    profile = {
        "email_hash": eh,
        "pin_hash": hash_pin(pin, eh),
        "name": (name or "").strip()[:30],
        "patient_memory": {"symptoms": [], "conditions": [], "medications": []},
        "language": "English",
        "created_at": firestore.SERVER_TIMESTAMP if FIREBASE_ACTIVE else now_str,
        "last_visit": firestore.SERVER_TIMESTAMP if FIREBASE_ACTIVE else now_str,
        "visit_count": 1,
    }
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        if "profiles" not in db:
            db["profiles"] = {}
        db["profiles"][eh] = profile
        _write_local_db(db)
        return profile
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
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        prof = db.get("profiles", {}).get(eh)
        if prof:
            prof["last_visit"] = datetime.now().isoformat()
            prof["visit_count"] = prof.get("visit_count", 0) + 1
            db["profiles"][eh] = prof
            _write_local_db(db)
            profile = prof
        profile["email_hash"] = eh
        return profile, "ok"
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
    if not email_hash:
        return
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        prof = db.get("profiles", {}).get(email_hash)
        if not prof:
            return
        prof["last_visit"] = datetime.now().isoformat()
        if patient_memory is not None:
            prof["patient_memory"] = patient_memory
        if name is not None:
            prof["name"] = (name or "").strip()[:30]
        if language is not None:
            prof["language"] = language
        if messages is not None:
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
            prof["messages"] = trimmed
        db["profiles"][email_hash] = prof
        _write_local_db(db)
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
            return (t[:25] + "…") if len(t) > 25 else t
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
        print("AI title generation failed:", e)
        return derive_chat_title(messages)

def list_conversations(email_hash, limit=30):
    if not email_hash:
        return []
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        user_convs = db.get("conversations", {}).get(email_hash, {})
        sorted_convs = sorted(
            user_convs.items(),
            key=lambda x: x[1].get("last_updated", ""),
            reverse=True
        )
        out = []
        for cid, data in sorted_convs[:limit]:
            out.append({
                "id": cid,
                "title": data.get("title", "Chat"),
                "message_count": data.get("message_count", 0),
                "last_updated": data.get("last_updated"),
                "first_user_msg": data.get("first_user_msg", ""),
                "last_assistant_msg": data.get("last_assistant_msg", ""),
            })
        return out
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
        print("list_conversations failed:", e)
        return []

def load_conversation(email_hash, conv_id):
    if not email_hash or not conv_id:
        return None
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        return db.get("conversations", {}).get(email_hash, {}).get(conv_id)
    try:
        doc = (firestore_db.collection("medichat_profiles")
               .document(email_hash)
               .collection("conversations")
               .document(conv_id).get())
        if not doc.exists:
            return None
        return doc.to_dict()
    except Exception as e:
        print("load_conversation failed:", e)
        return None

def save_conversation(email_hash, conv_id, messages):
    if not email_hash:
        return None
    trimmed = _trim_messages_for_storage(messages)
    msg_count = len(trimmed)
    # Lightweight summary fields for cross-chat context (avoids re-reading the full doc).
    _first_user = next((m.get("content", "") for m in trimmed if m.get("role") == "user"), "")
    _last_asst = next((m.get("content", "") for m in reversed(trimmed) if m.get("role") == "assistant"), "")
    now_str = datetime.now().isoformat()
    payload = {
        "messages": trimmed,
        "message_count": msg_count,
        "first_user_msg": (_first_user or "")[:240],
        "last_assistant_msg": (_last_asst or "")[:320],
        "last_updated": firestore.SERVER_TIMESTAMP if FIREBASE_ACTIVE else now_str,
    }
    # Title strategy: generate AI summary immediately on first save if Claude is active.
    if not conv_id:
        if CLAUDE_ACTIVE:
            payload["title"] = generate_ai_chat_title(trimmed)
        else:
            payload["title"] = derive_chat_title(trimmed)
    elif msg_count == 4:
        payload["title"] = generate_ai_chat_title(trimmed)

    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        if "conversations" not in db:
            db["conversations"] = {}
        if email_hash not in db["conversations"]:
            db["conversations"][email_hash] = {}
        if not conv_id:
            conv_id = str(_uuid.uuid4())[:16]
            payload["created_at"] = now_str
        else:
            existing = db["conversations"][email_hash].get(conv_id, {})
            existing.update(payload)
            payload = existing
        db["conversations"][email_hash][conv_id] = payload
        _write_local_db(db)
        return conv_id

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
        print("save_conversation failed:", e)
        return None

def delete_conversation(email_hash, conv_id):
    if not email_hash or not conv_id:
        return False
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        user_convs = db.get("conversations", {}).get(email_hash, {})
        if conv_id in user_convs:
            del user_convs[conv_id]
            _write_local_db(db)
            return True
        return False
    try:
        (firestore_db.collection("medichat_profiles")
         .document(email_hash)
         .collection("conversations")
         .document(conv_id).delete())
        return True
    except Exception as e:
        print("delete_conversation failed:", e)
        return False


# ── Generic per-user data store (Firestore for auth, session for guest) ─
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
    if not (st.session_state.get("is_authenticated") and st.session_state.get("user_email_hash")):
        return None
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        return db.get("profiles", {}).get(st.session_state.user_email_hash, {})
    try:
        snap = firestore_db.collection("medichat_profiles").document(st.session_state.user_email_hash).get()
        return snap.to_dict() or {} if snap.exists else {}
    except Exception as e:
        print("get_user_doc failed:", e)
        return {}

def update_user_doc(updates):
    """Patch the signed-in user's profile doc."""
    if not (st.session_state.get("is_authenticated") and st.session_state.get("user_email_hash")):
        return False
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        prof = db.get("profiles", {}).get(st.session_state.user_email_hash)
        if prof is None:
            return False
        # deep update / merge updates
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(prof.get(k), dict):
                prof[k].update(v)
            else:
                prof[k] = v
        db["profiles"][st.session_state.user_email_hash] = prof
        _write_local_db(db)
        return True
    try:
        firestore_db.collection("medichat_profiles").document(st.session_state.user_email_hash).set(updates, merge=True)
        return True
    except Exception as e:
        print("update_user_doc failed:", e)
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

# ── Health Records (file metadata only) ──────────────────────────────
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
    raw = all_dm.get(date_key, {})
    return {**DAILY_METRIC_DEFAULTS, **raw}

def heart_rate_status(bpm):
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
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        if "queries" not in db:
            db["queries"] = []
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
            "timestamp": datetime.now().isoformat(),
        }
        db["queries"].append(safe_data)
        _write_local_db(db)
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
    if not FIREBASE_ACTIVE:
        db = _read_local_db()
        queries = db.get("queries", [])
        sorted_queries = sorted(
            queries,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        return sorted_queries[:limit]
    try:
        docs = firestore_db.collection("medichat_queries").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        print("Firestore read failed:", e)
        return []

