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

st.set_page_config(
    page_title="MediChat — Your Health Assistant",
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

    .main .block-container {
        padding: 1.5rem 2rem 2rem 2rem;
        max-width: 820px;
    }

    /* Header */
    .header-card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        display: flex;
        align-items: center;
        gap: 1.2rem;
        border: 1px solid rgba(255,255,255,0.8);
    }
    .header-icon {
        font-size: 3rem;
        line-height: 1;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f766e;
        margin: 0;
        line-height: 1.1;
    }
    .header-subtitle {
        color: #64748b;
        font-size: 0.88rem;
        margin: 0.2rem 0 0 0;
        font-weight: 400;
    }

    /* Stats row */
    .stats-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    .stat-pill {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 50px;
        padding: 0.35rem 0.9rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #475569;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .stat-pill.green { color: #0f766e; border-color: #99f6e4; background: #f0fdfa; }
    .stat-pill.blue { color: #0369a1; border-color: #bae6fd; background: #f0f9ff; }
    .stat-pill.purple { color: #7c3aed; border-color: #ddd6fe; background: #faf5ff; }

    /* Disclaimer */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 0.65rem 1rem;
        color: #92400e;
        font-size: 0.78rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Welcome card */
    .welcome-card {
        background: white;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        margin: 0.5rem 0 1rem 0;
        border: 1px solid rgba(255,255,255,0.9);
    }
    .welcome-emoji { font-size: 3.5rem; margin-bottom: 0.8rem; }
    .welcome-title { font-size: 1.4rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; }
    .welcome-text { color: #64748b; font-size: 0.9rem; line-height: 1.6; margin-bottom: 1.2rem; }
    .chip-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; }
    .chip {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 50px;
        padding: 0.35rem 0.9rem;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Chat messages */
    .msg-wrap { margin: 0.7rem 0; }

    .user-wrap {
        display: flex;
        justify-content: flex-end;
        align-items: flex-end;
        gap: 0.5rem;
        margin: 0.7rem 0;
    }
    .bot-wrap {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 0.5rem;
        margin: 0.7rem 0;
    }

    .av {
        width: 34px; height: 34px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
    }
    .av-user { background: linear-gradient(135deg, #0d9488, #059669); }
    .av-bot {
        background: white;
        border: 2px solid #99f6e4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .user-bubble {
        background: linear-gradient(135deg, #0d9488, #059669);
        color: white;
        padding: 0.8rem 1.1rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 75%;
        font-size: 0.92rem;
        line-height: 1.55;
        box-shadow: 0 4px 16px rgba(13,148,136,0.2);
    }

    .bot-bubble {
        background: white;
        border: 1px solid #e2e8f0;
        color: #1e293b;
        padding: 0.8rem 1.1rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 78%;
        font-size: 0.92rem;
        line-height: 1.65;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .bot-bubble strong { color: #0f766e; }
    .bot-bubble ul { padding-left: 1.2rem; margin: 0.4rem 0; }
    .bot-bubble li { margin-bottom: 0.3rem; color: #334155; }

    .bot-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #0f766e;
        margin-bottom: 0.3rem;
        margin-left: 42px;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    .image-tag {
        background: #faf5ff;
        border: 1px solid #ddd6fe;
        border-radius: 10px;
        padding: 0.35rem 0.75rem;
        color: #7c3aed;
        font-size: 0.75rem;
        text-align: center;
        margin-bottom: 0.3rem;
        display: inline-block;
    }

    /* Input card */
    .input-card {
        background: white;
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        margin-top: 1rem;
        border: 1px solid #e2e8f0;
    }

    .stTextInput > div > div > input {
        background: #f8fafc !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0d9488 !important;
        background: white !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.1) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0d9488, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(13,148,136,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(13,148,136,0.35) !important;
    }

    .stFileUploader > div {
        background: #f8fafc !important;
        border: 1.5px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }
    .stFileUploader > div:hover {
        border-color: #0d9488 !important;
        background: #f0fdfa !important;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .clear-btn-wrap { text-align: center; margin-top: 0.8rem; }

    div[data-testid="stMarkdownContainer"] p { color: #334155; }
    div[data-testid="column"] { padding: 0 0.3rem !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: white !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * { color: #1e293b !important; }

    .sb-section { margin-bottom: 1.2rem; }
    .sb-title { font-size: 0.7rem; font-weight: 700; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
    .sb-stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .sb-stat-num { font-size: 1.4rem; font-weight: 800; color: #0f766e !important; line-height: 1; }
    .sb-stat-label { font-size: 0.65rem; color: #94a3b8 !important; font-weight: 500; margin-top: 0.1rem; }

    .sb-feature {
        display: flex; align-items: center; gap: 0.5rem;
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.45rem 0.7rem;
        margin-bottom: 0.3rem;
    }
    .sb-feature-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .sb-feature-name { font-size: 0.75rem; font-weight: 600; color: #334155 !important; }
    .sb-feature-status { font-size: 0.65rem; color: #22c55e !important; margin-left: auto; font-weight: 600; }

    .sb-tip {
        font-size: 0.73rem;
        color: #64748b !important;
        padding: 0.3rem 0;
        border-bottom: 1px solid #f1f5f9;
        line-height: 1.4;
        cursor: pointer;
    }
    .sb-tip:hover { color: #0f766e !important; }

    .sb-footer {
        font-size: 0.65rem;
        color: #cbd5e1 !important;
        text-align: center;
        padding-top: 1rem;
        border-top: 1px solid #f1f5f9;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ── API Setup ─────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("⚠️ API key not found. Please check your secrets configuration.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ── RAG System ────────────────────────────────────────────────────────
@st.cache_resource
def load_rag_system():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Dataset 1 — PubMedQA (medical research papers)
    pubmed = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:500]")
    pubmed_docs = []
    for item in pubmed:
        pubmed_docs.append(
            f"[PubMed Research]\nQuestion: {item['question']}\nAnswer: {item['long_answer']}"
        )

    # Dataset 2 — MedDialog (real doctor-patient conversations)
    meddialog = load_dataset("BinKhoaLe1812/MedDialog-EN-100k", split="train[:500]")
    dialog_docs = []
    for item in meddialog:
        dialog_docs.append(
            f"[Doctor-Patient Conversation]\nPatient: {item['input']}\nDoctor: {item['output']}"
        )

    # Combine both datasets
    documents = pubmed_docs + dialog_docs

    # Build FAISS index
    embeddings = embedder.encode(documents)
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings.astype('float32'))
    return embedder, idx, documents

with st.spinner("⏳ Loading MediChat... just a moment!"):
    embedder, index, documents = load_rag_system()

def encode_image(f):
    img = Image.open(f)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def medichat_rag(question, history):
    emb = embedder.encode([question]).astype('float32')
    _, idxs = index.search(emb, k=3)
    context = "\n\n---\n\n".join([documents[i] for i in idxs[0]])
    msgs = [{
        "role": "system",
        "content": (
            "You are MediChat, a warm, friendly, and professional health assistant. "
            "You help everyday people understand medical topics clearly and compassionately. "
            "Use simple language — avoid heavy jargon unless you explain it. "
            "Use the following real PubMed research to support your answer. "
            "Format your response with **bold** for key terms and bullet points where helpful. "
            "Always end with a warm reminder to consult a doctor for personal health concerns.\n\n"
            "PUBMED RESEARCH CONTEXT:\n" + context
        )
    }]
    for m in history:
        if m.get("type") == "text":
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": question})
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        temperature=0.6,
        max_tokens=1024
    )
    return r.choices[0].message.content

def medichat_vision(question, b64):
    prompt = question.strip() if question.strip() else "Please analyse this medical image and describe what you observe clearly."
    r = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are MediChat, a warm and professional clinical AI assistant. "
                        "Analyse this medical image carefully. "
                        "Provide: **Clinical Observations**, **Possible Conditions**, and **Recommendations**. "
                        "Use simple, compassionate language that a patient can understand. "
                        "Always remind the user to consult a qualified doctor.\n\n"
                        f"User's question: {prompt}"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                }
            ]
        }],
        temperature=0.5,
        max_tokens=1024
    )
    return r.choices[0].message.content

# ── Session State ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qcount" not in st.session_state:
    st.session_state.qcount = 0

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediChat")
    st.markdown("---")

    st.markdown('<div class="sb-title">Session Stats</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="sb-stat-card">
            <div class="sb-stat-num">{st.session_state.qcount}</div>
            <div class="sb-stat-label">Questions Asked</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="sb-stat-card">
            <div class="sb-stat-num">300</div>
            <div class="sb-stat-label">PubMed Docs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-title">Active Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-feature">
        <div class="sb-feature-dot" style="background:#0d9488;"></div>
        <div class="sb-feature-name">RAG Pipeline</div>
        <div class="sb-feature-status">● Live</div>
    </div>
    <div class="sb-feature">
        <div class="sb-feature-dot" style="background:#7c3aed;"></div>
        <div class="sb-feature-name">Vision AI</div>
        <div class="sb-feature-status">● Live</div>
    </div>
    <div class="sb-feature">
        <div class="sb-feature-dot" style="background:#0369a1;"></div>
        <div class="sb-feature-name">PubMed Search</div>
        <div class="sb-feature-status">● Live</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-title">💡 Try Asking</div>', unsafe_allow_html=True)
    tips = [
        "What causes high blood pressure?",
        "How does diabetes affect the body?",
        "What are signs of a healthy heart?",
        "How can I improve my sleep?",
        "What foods help reduce inflammation?",
    ]
    for tip in tips:
        st.markdown(f'<div class="sb-tip">→ {tip}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.qcount = 0
        st.rerun()

    st.markdown("""
    <div class="sb-footer">
        MediChat v1.0<br>
        ICT654 — Group 7<br>
        SISTC Melbourne, 2026
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="header-card">
    <div class="header-icon">🏥</div>
    <div>
        <div class="header-title">MediChat</div>
        <div class="header-subtitle">Your friendly AI health assistant — powered by real medical research</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats pills
st.markdown(f"""
<div class="stats-row">
    <span class="stat-pill green">🔬 RAG Active — PubMed Grounded</span>
    <span class="stat-pill purple">👁️ Vision Active — Image Analysis</span>
    <span class="stat-pill blue">📚 1000 Medical Documents</span></div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ MediChat provides general health information only — not a substitute for professional medical advice.
    Always consult a qualified doctor for personal health concerns.
</div>
""", unsafe_allow_html=True)

# Chat area
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-emoji">👋</div>
        <div class="welcome-title">Hello! How can I help you today?</div>
        <div class="welcome-text">
            I'm MediChat — your friendly AI health assistant.<br>
            Ask me anything about health, symptoms, or medications.<br>
            You can also upload a medical image for AI-powered analysis!
        </div>
        <div class="chip-row">
            <span class="chip">💊 Medications</span>
            <span class="chip">🫀 Heart Health</span>
            <span class="chip">🧬 Conditions</span>
            <span class="chip">🥗 Nutrition</span>
            <span class="chip">🧠 Mental Health</span>
            <span class="chip">🦠 Infections</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            if msg.get("type") == "image":
                st.markdown('<span class="image-tag">🖼️ Medical image uploaded for analysis</span>', unsafe_allow_html=True)
                if msg.get("content"):
                    st.markdown(f"""
                    <div class="user-wrap">
                        <div class="user-bubble">{msg["content"]}</div>
                        <div class="av av-user">👤</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="user-wrap">
                    <div class="user-bubble">{msg["content"]}</div>
                    <div class="av av-user">👤</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="bot-label">🏥 MediChat</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="bot-wrap">
                <div class="av av-bot">🏥</div>
                <div class="bot-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# Input card
st.markdown('<div class="input-card">', unsafe_allow_html=True)

st.markdown('<div class="section-label">📎 Upload a medical image (optional)</div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Upload an X-ray, skin photo, scan, or any medical image for AI analysis"
)

if uploaded_image:
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.image(uploaded_image, caption="✅ Image ready for analysis", use_column_width=True)

st.markdown('<div class="section-label" style="margin-top:0.8rem;">💬 Your question</div>', unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "",
        placeholder="Type your health question here... e.g. What causes headaches?",
        label_visibility="collapsed"
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.form_submit_button("Send to MediChat 💬")

st.markdown('</div>', unsafe_allow_html=True)

# Handle input
if submit and (user_input.strip() or uploaded_image):
    st.session_state.qcount += 1

    if uploaded_image:
        st.session_state.messages.append({
            "role": "user", "type": "image", "content": user_input.strip()
        })
        with st.spinner("🔍 Analysing your image..."):
            uploaded_image.seek(0)
            reply = medichat_vision(user_input, encode_image(uploaded_image))
    else:
        st.session_state.messages.append({
            "role": "user", "type": "text", "content": user_input.strip()
        })
        with st.spinner("🔬 Searching medical research..."):
            reply = medichat_rag(user_input, st.session_state.messages[:-1])

    st.session_state.messages.append({
        "role": "assistant", "type": "text", "content": reply
    })
    st.rerun()
