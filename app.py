import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import os
import base64
from PIL import Image
import io
from datetime import datetime

st.set_page_config(
    page_title="MediChat - Your Health Assistant",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(160deg, #f0f9ff 0%, #e0f2fe 40%, #f0fdf4 100%);
        min-height: 100vh;
    }
    .main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 820px; }

    .header-card {
        background: white; border-radius: 20px; padding: 1.4rem 2rem;
        margin-bottom: 1rem; box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        display: flex; align-items: center; gap: 1.2rem;
        border: 1px solid rgba(255,255,255,0.8);
    }
    .header-icon { font-size: 2.5rem; line-height: 1; }
    .header-title { font-size: 1.8rem; font-weight: 800; color: #0f766e; margin: 0; }
    .header-subtitle { color: #64748b; font-size: 0.85rem; margin: 0.2rem 0 0 0; }

    .stats-row { display: flex; gap: 0.6rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
    .stat-pill {
        background: white; border: 1px solid #e2e8f0; border-radius: 50px;
        padding: 0.3rem 0.8rem; font-size: 0.72rem; font-weight: 600;
        color: #475569; display: inline-flex; align-items: center; gap: 0.3rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .stat-pill.green { color: #0f766e; border-color: #99f6e4; background: #f0fdfa; }
    .stat-pill.blue { color: #0369a1; border-color: #bae6fd; background: #f0f9ff; }
    .stat-pill.purple { color: #7c3aed; border-color: #ddd6fe; background: #faf5ff; }
    .stat-pill.orange { color: #c2410c; border-color: #fed7aa; background: #fff7ed; }

    .disclaimer {
        background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px;
        padding: 0.6rem 1rem; color: #92400e; font-size: 0.76rem;
        margin-bottom: 0.8rem; text-align: center;
    }

    .welcome-card {
        background: white; border-radius: 20px; padding: 2.2rem 2rem;
        text-align: center; box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        margin: 0.5rem 0 1rem 0; border: 1px solid rgba(255,255,255,0.9);
    }
    .welcome-emoji { font-size: 3rem; margin-bottom: 0.7rem; }
    .welcome-title { font-size: 1.3rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem; }
    .welcome-text { color: #64748b; font-size: 0.87rem; line-height: 1.6; margin-bottom: 1rem; }
    .chip-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.4rem; }
    .chip {
        background: #f0fdf4; border: 1px solid #86efac; border-radius: 50px;
        padding: 0.3rem 0.8rem; color: #166534; font-size: 0.76rem; font-weight: 500;
    }

    .assessment-card {
        background: white; border-radius: 20px; padding: 1.5rem 1.8rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06); margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    .assessment-title { font-size: 1rem; font-weight: 700; color: #0f766e; margin-bottom: 0.3rem; }
    .assessment-subtitle { font-size: 0.78rem; color: #64748b; margin-bottom: 1rem; }

    .progress-bar-wrap {
        background: #f1f5f9; border-radius: 50px; height: 6px;
        margin-bottom: 1.2rem; overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #0d9488, #059669);
        border-radius: 50px;
    }
    .progress-label {
        font-size: 0.7rem; color: #94a3b8; margin-bottom: 0.3rem;
        display: flex; justify-content: space-between;
    }

    .question-bubble {
        background: linear-gradient(135deg, #f0fdfa, #ecfdf5);
        border: 1px solid #99f6e4;
        border-radius: 16px 16px 16px 4px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #134e4a;
        font-weight: 500;
        line-height: 1.5;
    }

    .user-wrap { display: flex; justify-content: flex-end; align-items: flex-end; gap: 0.5rem; margin: 0.6rem 0; }
    .bot-wrap { display: flex; justify-content: flex-start; align-items: flex-start; gap: 0.5rem; margin: 0.6rem 0; }
    .av { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; flex-shrink: 0; }
    .av-user { background: linear-gradient(135deg, #0d9488, #059669); }
    .av-bot { background: white; border: 2px solid #99f6e4; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .user-bubble {
        background: linear-gradient(135deg, #0d9488, #059669); color: white;
        padding: 0.75rem 1rem; border-radius: 16px 16px 4px 16px;
        max-width: 75%; font-size: 0.9rem; line-height: 1.5;
        box-shadow: 0 4px 16px rgba(13,148,136,0.2);
    }
    .bot-bubble {
        background: white; border: 1px solid #e2e8f0; color: #1e293b;
        padding: 0.75rem 1rem; border-radius: 16px 16px 16px 4px;
        max-width: 78%; font-size: 0.9rem; line-height: 1.65;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .bot-bubble strong { color: #0f766e; }
    .bot-bubble ul { padding-left: 1.2rem; margin: 0.4rem 0; }
    .bot-bubble li { margin-bottom: 0.3rem; color: #334155; }
    .bot-label { font-size: 0.7rem; font-weight: 700; color: #0f766e; margin-bottom: 0.25rem; margin-left: 40px; }

    .image-tag {
        background: #faf5ff; border: 1px solid #ddd6fe; border-radius: 10px;
        padding: 0.3rem 0.7rem; color: #7c3aed; font-size: 0.73rem;
        text-align: center; margin-bottom: 0.3rem; display: inline-block;
    }

    .memory-card {
        background: #f0fdf4; border: 1px solid #86efac; border-radius: 12px;
        padding: 0.6rem 1rem; margin-bottom: 0.8rem; font-size: 0.76rem; color: #166534;
    }
    .memory-title { font-weight: 700; margin-bottom: 0.25rem; font-size: 0.78rem; }

    .input-card {
        background: white; border-radius: 20px; padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06); margin-top: 0.8rem;
        border: 1px solid #e2e8f0;
    }
    .stTextInput > div > div > input {
        background: #f8fafc !important; border: 1.5px solid #e2e8f0 !important;
        border-radius: 12px !important; color: #1e293b !important;
        padding: 0.75rem 1rem !important; font-size: 0.9rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0d9488 !important; background: white !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.1) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #0d9488, #059669) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 0.65rem 1.3rem !important; font-weight: 600 !important;
        font-size: 0.88rem !important; transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(13,148,136,0.25) !important;
    }
    .stButton > button:hover { transform: translateY(-1px) !important; }
    .stFileUploader > div {
        background: #f8fafc !important; border: 1.5px dashed #cbd5e1 !important;
        border-radius: 12px !important;
    }
    .section-label { font-size: 0.73rem; font-weight: 600; color: #64748b; margin-bottom: 0.35rem; text-transform: uppercase; letter-spacing: 0.06em; }
    div[data-testid="stMarkdownContainer"] p { color: #334155; }
    div[data-testid="column"] { padding: 0 0.25rem !important; }

    section[data-testid="stSidebar"] { background: white !important; border-right: 1px solid #e2e8f0 !important; }
    section[data-testid="stSidebar"] * { color: #1e293b !important; }
    .sb-title { font-size: 0.68rem; font-weight: 700; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
    .sb-stat-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.55rem 0.75rem; margin-bottom: 0.35rem; }
    .sb-stat-num { font-size: 1.3rem; font-weight: 800; color: #0f766e !important; line-height: 1; }
    .sb-stat-label { font-size: 0.63rem; color: #94a3b8 !important; font-weight: 500; margin-top: 0.1rem; }
    .sb-feature { display: flex; align-items: center; gap: 0.5rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.4rem 0.65rem; margin-bottom: 0.3rem; }
    .sb-feature-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .sb-feature-name { font-size: 0.73rem; font-weight: 600; color: #334155 !important; }
    .sb-feature-status { font-size: 0.63rem; color: #22c55e !important; margin-left: auto; font-weight: 600; }
    .sb-tip { font-size: 0.71rem; color: #64748b !important; padding: 0.28rem 0; border-bottom: 1px solid #f1f5f9; line-height: 1.4; }
    .sb-memory-item { font-size: 0.68rem; color: #0f766e !important; padding: 0.22rem 0; border-bottom: 1px solid #f0fdf4; }
    .sb-footer { font-size: 0.63rem; color: #cbd5e1 !important; text-align: center; padding-top: 0.8rem; border-top: 1px solid #f1f5f9; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── API Setup ─────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("API key not found. Please check your secrets configuration.")
    st.stop()
groq_client = Groq(api_key=GROQ_API_KEY)

# ── RAG System ────────────────────────────────────────────────────────
@st.cache_resource
def load_rag_system():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    pubmed = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:500]")
    pubmed_docs = [
        "[PubMed Research]\nQuestion: " + i["question"] + "\nAnswer: " + i["long_answer"]
        for i in pubmed
    ]
    meddialog = load_dataset("BinKhoaLe1812/MedDialog-EN-100k", split="train[:500]")
    dialog_docs = [
        "[Doctor-Patient Conversation]\nPatient: " + i["input"] + "\nDoctor: " + i["output"]
        for i in meddialog
    ]
    docs = pubmed_docs + dialog_docs
    embeddings = embedder.encode(docs)
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings.astype("float32"))
    return embedder, idx, docs

with st.spinner("Loading MediChat knowledge base... just a moment!"):
    embedder, index, documents = load_rag_system()

def encode_image(f):
    img = Image.open(f)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Memory ────────────────────────────────────────────────────────────
def extract_patient_memory(messages):
    memory = {"symptoms": [], "conditions": [], "medications": []}
    symptom_phrases = [
        "i have", "i feel", "i am feeling", "i've been feeling", "i'm experiencing",
        "i suffer from", "my back hurts", "my chest", "my stomach", "i feel pain",
        "feeling dizzy", "feeling nauseous", "i have a fever", "i have a cough",
        "i have a headache", "shortness of breath", "i've been tired", "i feel tired",
        "i have pain", "i feel sick", "i have been vomiting", "i have swelling", "i have a rash"
    ]
    condition_phrases = [
        "i have diabetes", "i have hypertension", "i have asthma", "i have cancer",
        "i am diabetic", "i am hypertensive", "i was diagnosed with",
        "i have high blood pressure", "i have low blood pressure",
        "i have depression", "i have anxiety", "i have heart", "i have kidney"
    ]
    medication_phrases = [
        "i am taking", "i take", "i was prescribed", "i'm on",
        "taking medication", "prescribed me", "i have an inhaler",
        "i take tablets", "i take pills"
    ]
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

# ── RAG Chat ──────────────────────────────────────────────────────────
def medichat_rag(question, all_messages):
    emb = embedder.encode([question]).astype("float32")
    _, idxs = index.search(emb, k=3)
    context = "\n\n---\n\n".join([documents[i] for i in idxs[0]])
    memory = extract_patient_memory(all_messages)
    memory_context = build_memory_context(memory)
    history = []
    for m in all_messages[-10:]:
        if m.get("type") == "text":
            history.append({"role": m["role"], "content": m["content"]})
    system = (
        "You are MediChat, a warm, friendly, and conversational health assistant. "
        "Talk naturally like a caring human. Keep responses clear and simple. "
        "STRICT RULES: "
        "1) NEVER mention anything the patient has NOT said in THIS conversation. "
        "2) Use the medical research below as background knowledge only. "
        "3) NEVER invent symptoms or history. "
        "4) Respond warmly to casual messages. "
        "5) Only reference earlier parts of THIS conversation if the patient actually said it.\n\n"
    )
    if memory_context:
        system += "WHAT THIS PATIENT HAS TOLD YOU:\n" + memory_context + "\n\n"
    system += "BACKGROUND MEDICAL KNOWLEDGE:\n" + context
    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}]
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=msgs, temperature=0.6, max_tokens=1024
    )
    return r.choices[0].message.content, memory

def medichat_vision(question, b64, all_messages):
    memory = extract_patient_memory(all_messages)
    memory_context = build_memory_context(memory)
    prompt = question.strip() if question.strip() else "Please analyse this medical image."
    memory_note = ("\n\nPatient context: " + memory_context) if memory_context else ""
    r = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "You are MediChat, a warm clinical AI assistant. "
                "Analyse this medical image. "
                "Provide: Clinical Observations, Possible Conditions, Recommendations. "
                "Use simple compassionate language. Always recommend consulting a doctor."
                + memory_note + "\n\nQuestion: " + prompt
            )},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}
        ]}],
        temperature=0.5, max_tokens=1024
    )
    return r.choices[0].message.content

# ── Assessment ────────────────────────────────────────────────────────
ASSESSMENT_STAGES = [
    {
        "key": "main_symptom",
        "icon": "What is your main symptom or health concern today?",
        "hint": "e.g. chest pain, headache, fever, cough...",
        "options": []
    },
    {
        "key": "duration",
        "icon": "How long have you been experiencing this?",
        "hint": "",
        "options": ["Just started today", "A few days", "About a week", "More than 2 weeks", "Over a month"]
    },
    {
        "key": "severity",
        "icon": "How severe is it on a scale of 1 to 10? (1 = mild, 10 = unbearable)",
        "hint": "",
        "options": ["1-2 (Mild)", "3-4 (Moderate)", "5-6 (Significant)", "7-8 (Severe)", "9-10 (Unbearable)"]
    },
    {
        "key": "pattern",
        "icon": "Is it constant or does it come and go?",
        "hint": "",
        "options": ["Constant, always there", "Comes and goes", "Getting worse over time", "Getting better", "Only happens sometimes"]
    },
    {
        "key": "other_symptoms",
        "icon": "Are you experiencing any other symptoms alongside this?",
        "hint": "e.g. nausea, dizziness, fever, fatigue... or type None",
        "options": ["No other symptoms", "Nausea or vomiting", "Fever or chills", "Dizziness", "Fatigue or weakness", "Other (type below)"]
    },
    {
        "key": "age",
        "icon": "How old are you?",
        "hint": "",
        "options": ["Under 18", "18-30", "31-45", "46-60", "61-75", "Over 75"]
    },
    {
        "key": "gender",
        "icon": "What is your biological sex? (helps with medical accuracy)",
        "hint": "",
        "options": ["Male", "Female", "Prefer not to say"]
    },
]

def generate_assessment_report(assessment_data):
    emb = embedder.encode([assessment_data.get("main_symptom", "")]).astype("float32")
    _, idxs = index.search(emb, k=5)
    context = "\n\n---\n\n".join([documents[i] for i in idxs[0]])

    prompt = (
        "You are an experienced clinical AI assistant helping a patient understand their symptoms.\n\n"
        "A patient completed a symptom assessment. Analyse their responses carefully and generate "
        "a medically sound, compassionate report.\n\n"
        "PATIENT RESPONSES:\n"
        "- Main symptom: " + assessment_data.get("main_symptom", "Not specified") + "\n"
        "- Duration: " + assessment_data.get("duration", "Not specified") + "\n"
        "- Severity: " + assessment_data.get("severity", "Not specified") + "\n"
        "- Pattern: " + assessment_data.get("pattern", "Not specified") + "\n"
        "- Other symptoms: " + assessment_data.get("other_symptoms", "None") + "\n"
        "- Age group: " + assessment_data.get("age", "Not specified") + "\n"
        "- Biological sex: " + assessment_data.get("gender", "Not specified") + "\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "1. If the main symptom appears to be a typo or unclear word, use clinical reasoning to "
        "interpret the most likely intended symptom based on context (age, sex, pattern, severity).\n"
        "2. Consider all symptoms together holistically — not in isolation.\n"
        "3. Provide realistic, evidence-based possible conditions — not rare or alarming ones unless severity warrants.\n"
        "4. Urgency must reflect the actual severity and pattern described.\n"
        "5. Next steps must be specific and actionable.\n\n"
        "MEDICAL RESEARCH CONTEXT (use to support your reasoning):\n" + context + "\n\n"
        "Respond in EXACTLY this format — no extra text:\n"
        "URGENCY: [one of: Self-care at home / See a doctor soon / Seek urgent care today / Go to emergency NOW]\n"
        "CONDITIONS: [condition 1] | [condition 2] | [condition 3]\n"
        "NEXT STEPS: [step 1] | [step 2] | [step 3]\n"
        "SUMMARY: [2-3 warm, clear sentences summarising the assessment and what the patient should do]\n"
    )

    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )
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

# ── Session State ─────────────────────────────────────────────────────
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

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## MediChat")
    st.markdown("---")
    st.markdown('<div class="sb-title">Session Stats</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="sb-stat-card"><div class="sb-stat-num">' +
            str(st.session_state.qcount) +
            '</div><div class="sb-stat-label">Questions</div></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="sb-stat-card"><div class="sb-stat-num">1000</div><div class="sb-stat-label">Medical Docs</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    mem = st.session_state.patient_memory
    has_memory = any([mem.get("symptoms"), mem.get("conditions"), mem.get("medications")])
    if has_memory:
        st.markdown('<div class="sb-title">Patient Memory</div>', unsafe_allow_html=True)
        if mem.get("symptoms"):
            st.markdown('<div class="sb-memory-item">Symptoms: ' + ", ".join(mem["symptoms"][:2]) + '</div>', unsafe_allow_html=True)
        if mem.get("conditions"):
            st.markdown('<div class="sb-memory-item">Conditions: ' + ", ".join(mem["conditions"][:2]) + '</div>', unsafe_allow_html=True)
        if mem.get("medications"):
            st.markdown('<div class="sb-memory-item">Medications: ' + ", ".join(mem["medications"][:2]) + '</div>', unsafe_allow_html=True)
        st.markdown("---")

    st.markdown('<div class="sb-title">Active Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-feature"><div class="sb-feature-dot" style="background:#0d9488;"></div><div class="sb-feature-name">RAG Pipeline</div><div class="sb-feature-status">Live</div></div>
    <div class="sb-feature"><div class="sb-feature-dot" style="background:#7c3aed;"></div><div class="sb-feature-name">Vision AI</div><div class="sb-feature-status">Live</div></div>
    <div class="sb-feature"><div class="sb-feature-dot" style="background:#0369a1;"></div><div class="sb-feature-name">Chat Memory</div><div class="sb-feature-status">Live</div></div>
    <div class="sb-feature"><div class="sb-feature-dot" style="background:#059669;"></div><div class="sb-feature-name">Symptom Check</div><div class="sb-feature-status">Live</div></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-title">Try Asking</div>', unsafe_allow_html=True)
    tips = [
        "What causes high blood pressure?",
        "I have chest pain and I am diabetic",
        "How does stress affect the heart?",
        "What foods reduce inflammation?",
        "I have been dizzy since yesterday"
    ]
    for tip in tips:
        st.markdown('<div class="sb-tip">- ' + tip + '</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div class="sb-footer">MediChat v3.0<br>ICT654 - Group 3 - SISTC 2026</div>',
        unsafe_allow_html=True
    )

# ── MAIN ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div class="header-icon">🏥</div>
    <div>
        <div class="header-title">MediChat</div>
        <div class="header-subtitle">Your AI health assistant — chat freely or do a guided symptom check</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode buttons
col_m1, col_m2 = st.columns(2)
with col_m1:
    chat_label = "💬 Free Chat  ✓" if st.session_state.mode == "chat" else "💬 Free Chat"
    if st.button(chat_label, use_container_width=True):
        st.session_state.mode = "chat"
        st.rerun()
with col_m2:
    assess_label = "🔍 Symptom Check  ✓" if st.session_state.mode == "assessment" else "🔍 Symptom Check"
    if st.button(assess_label, use_container_width=True):
        st.session_state.mode = "assessment"
        st.rerun()

st.markdown("""
<div class="stats-row">
    <span class="stat-pill green">RAG — PubMed + MedDialog</span>
    <span class="stat-pill purple">Vision AI Active</span>
    <span class="stat-pill blue">1000 Medical Docs</span>
    <span class="stat-pill orange">Memory Active</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    MediChat provides general health information only — not a substitute for professional medical advice.
    Always consult a qualified doctor for personal health concerns.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MODE 1 — FREE CHAT
# ══════════════════════════════════════════════════════════════════════
if st.session_state.mode == "chat":
    mem = st.session_state.patient_memory
    has_memory = any([mem.get("symptoms"), mem.get("conditions"), mem.get("medications")])
    if has_memory and st.session_state.messages:
        mem_parts = []
        if mem.get("symptoms"):
            mem_parts.append("Symptoms: " + ", ".join(mem["symptoms"]))
        if mem.get("conditions"):
            mem_parts.append("Conditions: " + ", ".join(mem["conditions"]))
        if mem.get("medications"):
            mem_parts.append("Medications: " + ", ".join(mem["medications"]))
        items_html = "".join(['<div>- ' + p + '</div>' for p in mem_parts])
        st.markdown(
            '<div class="memory-card"><div class="memory-title">MediChat remembers from this session:</div>' + items_html + '</div>',
            unsafe_allow_html=True
        )

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-emoji">👋</div>
            <div class="welcome-title">Hello! How can I help you today?</div>
            <div class="welcome-text">
                I am MediChat, your friendly AI health assistant.<br>
                I remember everything you tell me during our conversation.<br><br>
                Or switch to Symptom Check for a guided assessment like Ada Health!
            </div>
            <div class="chip-row">
                <span class="chip">Medications</span>
                <span class="chip">Heart Health</span>
                <span class="chip">Conditions</span>
                <span class="chip">Nutrition</span>
                <span class="chip">Mental Health</span>
                <span class="chip">Infections</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if msg.get("type") == "image":
                    st.markdown('<span class="image-tag">Medical image uploaded for analysis</span>', unsafe_allow_html=True)
                    if msg.get("content"):
                        st.markdown(
                            '<div class="user-wrap"><div class="user-bubble">' + msg["content"] + '</div><div class="av av-user">👤</div></div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        '<div class="user-wrap"><div class="user-bubble">' + msg["content"] + '</div><div class="av av-user">👤</div></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown('<div class="bot-label">MediChat</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="bot-wrap"><div class="av av-bot">🏥</div><div class="bot-bubble">' + msg["content"] + '</div></div>',
                    unsafe_allow_html=True
                )

    if st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;font-size:0.78rem;color:#64748b;margin-bottom:0.4rem;">Was this conversation helpful?</div>',
            unsafe_allow_html=True
        )
        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([2, 1, 0.5, 1, 2])
        with col_f2:
            if st.button("Yes", key="chat_helpful"):
                st.session_state.feedback["overall"] = "helpful"
                st.rerun()
        with col_f4:
            if st.button("No", key="chat_not_helpful"):
                st.session_state.feedback["overall"] = "not_helpful"
                st.rerun()
        overall = st.session_state.feedback.get("overall")
        if overall == "helpful":
            st.markdown(
                '<div style="text-align:center;font-size:0.76rem;color:#0f766e;margin-top:0.3rem;">Thank you! Glad MediChat was helpful.</div>',
                unsafe_allow_html=True
            )
        elif overall == "not_helpful":
            st.markdown(
                '<div style="text-align:center;font-size:0.76rem;color:#dc2626;margin-top:0.3rem;">Thank you for your feedback. We will keep improving!</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Upload a medical image (optional)</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="uploader_" + str(st.session_state.uploader_key)
    )
    if uploaded_image:
        cola, colb, colc = st.columns([1, 2, 1])
        with colb:
            st.image(uploaded_image, caption="Ready for analysis", use_column_width=True)
    st.markdown('<div class="section-label" style="margin-top:0.7rem;">Your question</div>', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("", placeholder="Type your health question here...", label_visibility="collapsed")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col2:
            submit = st.form_submit_button("Send to MediChat")
        with col3:
            clear = st.form_submit_button("Clear")
    st.markdown('</div>', unsafe_allow_html=True)

    if clear:
        st.session_state.messages = []
        st.session_state.qcount = 0
        st.session_state.feedback = {}
        st.session_state.patient_memory = {"symptoms": [], "conditions": [], "medications": []}
        st.session_state.uploader_key += 1
        st.rerun()

    if submit and (user_input.strip() or uploaded_image):
        st.session_state.qcount += 1
        if uploaded_image:
            st.session_state.messages.append({"role": "user", "type": "image", "content": user_input.strip()})
            with st.spinner("Analysing your image..."):
                uploaded_image.seek(0)
                reply = medichat_vision(user_input, encode_image(uploaded_image), st.session_state.messages)
        else:
            st.session_state.messages.append({"role": "user", "type": "text", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                reply, memory = medichat_rag(user_input, st.session_state.messages)
                st.session_state.patient_memory = memory
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# MODE 2 — SYMPTOM ASSESSMENT
# ══════════════════════════════════════════════════════════════════════
else:
    if st.session_state.assessment_complete and st.session_state.assessment_parsed:
        parsed = st.session_state.assessment_parsed
        data = st.session_state.assessment_data
        urgency = parsed.get("urgency", "See a doctor soon")
        urgency_lower = urgency.lower()
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        st.markdown("---")
        st.markdown("### MediChat Assessment Report")
        st.caption("Generated: " + report_date)

        if "emergency" in urgency_lower or "now" in urgency_lower:
            st.error("URGENCY: " + urgency)
        elif "urgent" in urgency_lower or "today" in urgency_lower:
            st.warning("URGENCY: " + urgency)
        else:
            st.success("URGENCY: " + urgency)

        st.markdown("---")
        st.markdown("#### Symptoms Reported")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Main symptom:** " + data.get("main_symptom", ""))
            st.info("**Duration:** " + data.get("duration", ""))
            st.info("**Severity:** " + data.get("severity", ""))
        with col2:
            st.info("**Pattern:** " + data.get("pattern", ""))
            st.info("**Age:** " + data.get("age", ""))
            st.info("**Sex:** " + data.get("gender", ""))

        other = data.get("other_symptoms", "")
        if other and other.lower() not in ["no", "none", "n/a", "no other symptoms"]:
            st.info("**Other symptoms:** " + other)

        st.markdown("---")
        st.markdown("#### Possible Conditions")
        conditions = parsed.get("conditions", [])
        if conditions:
            for c in conditions:
                if c.strip():
                    st.markdown("- " + c.strip())
        else:
            st.markdown("- Please consult a doctor for a proper assessment.")

        st.markdown("---")
        st.markdown("#### What To Do Next")
        steps = parsed.get("next_steps", [])
        if steps:
            for i, s in enumerate(steps, 1):
                if s.strip():
                    st.markdown("**" + str(i) + ".** " + s.strip())
        else:
            st.markdown("**1.** Consult a qualified healthcare professional.")

        st.markdown("---")
        st.markdown("#### MediChat Summary")
        summary = parsed.get("summary", "")
        if summary:
            st.markdown(
                '<div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:12px;padding:1rem;font-size:0.92rem;color:#134e4a;line-height:1.6;">' +
                summary +
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.warning("This assessment is for information only and is NOT a medical diagnosis. Please consult a qualified healthcare professional for proper evaluation and treatment.")

        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Start New Assessment", use_container_width=True):
                st.session_state.assessment_stage = 0
                st.session_state.assessment_data = {}
                st.session_state.assessment_complete = False
                st.session_state.assessment_report = None
                st.session_state.assessment_parsed = None
                st.rerun()
        with col_r2:
            if st.button("Switch to Free Chat", use_container_width=True):
                st.session_state.mode = "chat"
                st.rerun()

    else:
        stage = st.session_state.assessment_stage
        total = len(ASSESSMENT_STAGES)
        progress = int((stage / total) * 100)

        st.markdown(
            '<div class="assessment-card">'
            '<div class="assessment-title">Symptom Assessment</div>'
            '<div class="assessment-subtitle">Answer a few quick questions and MediChat will generate a personalised health assessment.</div>'
            '<div class="progress-label"><span>Step ' + str(stage + 1) + ' of ' + str(total) + '</span><span>' + str(progress) + '% complete</span></div>'
            '<div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:' + str(progress) + '%;"></div></div>'
            '</div>',
            unsafe_allow_html=True
        )

        if st.session_state.assessment_data:
            with st.expander("Your answers so far", expanded=False):
                for k, v in st.session_state.assessment_data.items():
                    label = k.replace("_", " ").title()
                    st.markdown("**" + label + ":** " + v)

        if stage < total:
            current = ASSESSMENT_STAGES[stage]

            st.markdown(
                '<div class="question-bubble">' + current["icon"] + '</div>',
                unsafe_allow_html=True
            )

            if current["hint"]:
                st.caption(current["hint"])

            if current["options"]:
                st.markdown("**Quick select:**")
                num_cols = min(len(current["options"]), 3)
                option_cols = st.columns(num_cols)
                for i, opt in enumerate(current["options"]):
                    with option_cols[i % num_cols]:
                        if st.button(opt, key="opt_" + str(stage) + "_" + str(i), use_container_width=True):
                            st.session_state.assessment_data[current["key"]] = opt
                            st.session_state.assessment_stage += 1
                            if st.session_state.assessment_stage >= total:
                                with st.spinner("Generating your personalised assessment..."):
                                    report = generate_assessment_report(st.session_state.assessment_data)
                                    st.session_state.assessment_report = report
                                    st.session_state.assessment_parsed = parse_report(report)
                                    st.session_state.assessment_complete = True
                            st.rerun()

            with st.form(key="assessment_form_" + str(stage), clear_on_submit=True):
                typed = st.text_input("", placeholder="Or type your answer here...", label_visibility="collapsed")
                col_a1, col_a2, col_a3 = st.columns([2, 2, 1])
                with col_a2:
                    next_btn = st.form_submit_button("Next")
                with col_a3:
                    cancel_btn = st.form_submit_button("Cancel")

            if next_btn and typed.strip():
                st.session_state.assessment_data[current["key"]] = typed.strip()
                st.session_state.assessment_stage += 1
                if st.session_state.assessment_stage >= total:
                    with st.spinner("Generating your personalised assessment..."):
                        report = generate_assessment_report(st.session_state.assessment_data)
                        st.session_state.assessment_report = report
                        st.session_state.assessment_parsed = parse_report(report)
                        st.session_state.assessment_complete = True
                st.rerun()

            if cancel_btn:
                st.session_state.assessment_stage = 0
                st.session_state.assessment_data = {}
                st.session_state.mode = "chat"
                st.rerun()
