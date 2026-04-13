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
    page_title="MediChat — Clinical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #060d1a;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%);
        border-right: 1px solid rgba(0, 201, 167, 0.15);
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Main area */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 900px;
    }

    /* Header */
    .medichat-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, rgba(0,201,167,0.08), rgba(11,123,139,0.08));
        border: 1px solid rgba(0,201,167,0.2);
        border-radius: 16px;
        margin-bottom: 1.2rem;
    }

    .medichat-logo {
        font-size: 2.8rem;
    }

    .medichat-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c9a7, #0b7b8b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }

    .medichat-subtitle {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0;
        font-weight: 400;
    }

    /* Badges row */
    .badges-row {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }

    .badge {
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .badge-rag {
        background: rgba(0, 201, 167, 0.1);
        border: 1px solid rgba(0, 201, 167, 0.3);
        color: #00c9a7;
    }

    .badge-vision {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #a78bfa;
    }

    .badge-live {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(251, 191, 36, 0.06);
        border: 1px solid rgba(251, 191, 36, 0.2);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        color: #fbbf24;
        font-size: 0.78rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Chat messages */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem 0;
    }

    .user-row {
        display: flex;
        justify-content: flex-end;
        align-items: flex-end;
        gap: 0.6rem;
    }

    .bot-row {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 0.6rem;
    }

    .avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }

    .avatar-user {
        background: linear-gradient(135deg, #0b7b8b, #00897b);
    }

    .avatar-bot {
        background: linear-gradient(135deg, #1e3a5f, #0d2b55);
        border: 1px solid rgba(0,201,167,0.3);
    }

    .user-bubble {
        background: linear-gradient(135deg, #0b7b8b, #00897b);
        color: white;
        padding: 0.85rem 1.1rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 75%;
        font-size: 0.93rem;
        line-height: 1.55;
        box-shadow: 0 4px 20px rgba(0, 201, 167, 0.15);
    }

    .bot-bubble {
        background: linear-gradient(135deg, rgba(13,27,58,0.9), rgba(10,22,40,0.9));
        border: 1px solid rgba(255,255,255,0.08);
        color: #e2e8f0;
        padding: 0.85rem 1.1rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 80%;
        font-size: 0.93rem;
        line-height: 1.65;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .bot-bubble strong {
        color: #00c9a7;
    }

    .bot-bubble ul {
        padding-left: 1.2rem;
        margin: 0.4rem 0;
    }

    .bot-bubble li {
        margin-bottom: 0.3rem;
    }

    .image-tag {
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        color: #a78bfa;
        font-size: 0.78rem;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .welcome-card {
        background: linear-gradient(135deg, rgba(13,27,58,0.6), rgba(10,22,40,0.6));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
    }

    .welcome-title {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .welcome-text {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .suggestion-row {
        display: flex;
        gap: 0.6rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .suggestion {
        background: rgba(0,201,167,0.08);
        border: 1px solid rgba(0,201,167,0.2);
        border-radius: 20px;
        padding: 0.4rem 1rem;
        color: #00c9a7;
        font-size: 0.8rem;
        cursor: pointer;
    }

    /* Input area */
    .input-area {
        background: linear-gradient(135deg, rgba(13,27,58,0.8), rgba(10,22,40,0.8));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.93rem !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00c9a7 !important;
        box-shadow: 0 0 0 3px rgba(0,201,167,0.1) !important;
        background: rgba(0,201,167,0.04) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #475569 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00c9a7, #0b7b8b) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.93rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.02em !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(0,201,167,0.3) !important;
    }

    /* File uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(13,27,58,0.8), rgba(10,22,40,0.8));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stat-number {
        font-size: 1.6rem;
        font-weight: 800;
        color: #00c9a7;
        line-height: 1;
    }

    .stat-label {
        font-size: 0.7rem;
        color: #475569;
        font-weight: 500;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    div[data-testid="stMarkdownContainer"] p { color: #e2e8f0; }
    div[data-testid="column"] { padding: 0 0.3rem !important; }
</style>
""", unsafe_allow_html=True)

# ── API + RAG Setup ───────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("⚠️ API key not found. Please check your secrets configuration.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_rag_system():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:300]")
    documents = []
    for item in dataset:
        combined = f"Question: {item['question']}\nAnswer: {item['long_answer']}"
        documents.append(combined)
    embeddings = embedder.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return embedder, index, documents

with st.spinner("🔬 Loading MediChat knowledge base..."):
    embedder, index, documents = load_rag_system()

def encode_image(image_file):
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def medichat_rag(user_question, chat_history):
    question_embedding = embedder.encode([user_question]).astype('float32')
    distances, indices = index.search(question_embedding, k=3)
    relevant_docs = [documents[i] for i in indices[0]]
    context = "\n\n---\n\n".join(relevant_docs)
    messages = [{
        "role": "system",
        "content": (
            "You are MediChat, a professional and empathetic clinical AI assistant. "
            "Use the following real medical research context to answer accurately. "
            "Format your response clearly — use **bold** for key terms, bullet points for lists. "
            "Always recommend consulting a qualified doctor for personal medical advice.\n\n"
            "MEDICAL RESEARCH CONTEXT:\n" + context
        )
    }]
    for msg in chat_history:
        if msg.get("type") == "text":
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_question})
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def medichat_vision(user_question, image_b64):
    prompt = user_question if user_question.strip() else "Analyse this medical image and describe clinical observations in detail."
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are MediChat, a clinical AI assistant. "
                        "Analyse this medical image carefully. "
                        "Provide: 1) Clinical Observations, 2) Possible Differential Diagnoses, 3) Recommendations. "
                        "Use clear clinical language with **bold** headings. "
                        "Always remind the user to consult a healthcare professional.\n\n"
                        f"User question: {prompt}"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        }],
        temperature=0.5,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# ── Session State ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediChat")
    st.markdown("---")

    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.question_count}</div>
            <div class="stat-label">Questions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">300</div>
            <div class="stat-label">PubMed Docs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Active Features")
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:0.5rem;">
        <div class="badge badge-rag">🔬 RAG Pipeline</div>
        <div class="badge badge-vision">👁️ Vision AI</div>
        <div class="badge badge-live">🟢 Live</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Try Asking")
    suggestions = [
        "What causes hypertension?",
        "Symptoms of type 2 diabetes",
        "How does the immune system work?",
        "What is pneumonia?",
    ]
    for s in suggestions:
        st.markdown(f"- *{s}*")

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    <div style="color:#94a3b8;font-size:0.78rem;line-height:1.5;">
    MediChat provides general medical information only. Always consult a qualified healthcare professional for personal medical advice.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()

    st.markdown("""
    <div style="color:#334155;font-size:0.72rem;text-align:center;margin-top:1rem;">
    ICT654 — Group 7 — SISTC Melbourne
    </div>
    """, unsafe_allow_html=True)

# ── MAIN AREA ─────────────────────────────────────────────────────────
st.markdown("""
<div class="medichat-header">
    <div class="medichat-logo">🏥</div>
    <div>
        <div class="medichat-title">MediChat</div>
        <div class="medichat-subtitle">Clinical AI Assistant — Powered by LLMs & RAG Pipeline</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badges-row">
    <span class="badge badge-rag">🔬 RAG Active — Grounded in PubMed Research</span>
    <span class="badge badge-vision">👁️ Vision Active — Medical Image Analysis</span>
    <span class="badge badge-live">🟢 Live</span>
</div>
""", unsafe_allow_html=True)

# ── Chat Display ──────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">🏥</div>
        <div class="welcome-title">Welcome to MediChat</div>
        <div class="welcome-text">
            Your intelligent clinical AI assistant powered by real medical research.<br>
            Ask any medical question or upload a medical image for analysis.
        </div>
        <div class="suggestion-row">
            <span class="suggestion">💊 Drug interactions</span>
            <span class="suggestion">🫀 Heart conditions</span>
            <span class="suggestion">🧬 Genetics</span>
            <span class="suggestion">🦠 Infections</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            if msg.get("type") == "image":
                st.markdown('<div class="image-tag">🖼️ Medical image uploaded for analysis</div>', unsafe_allow_html=True)
                if msg.get("content"):
                    st.markdown(f"""
                    <div class="user-row">
                        <div class="user-bubble">{msg["content"]}</div>
                        <div class="avatar avatar-user">👤</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="user-row">
                    <div class="user-bubble">{msg["content"]}</div>
                    <div class="avatar avatar-user">👤</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-row">
                <div class="avatar avatar-bot">🏥</div>
                <div class="bot-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Input Area ────────────────────────────────────────────────────────
st.markdown('<div class="input-area">', unsafe_allow_html=True)

uploaded_image = st.file_uploader(
    "📎 Upload a medical image for analysis",
    type=["jpg", "jpeg", "png"],
    help="Supports X-rays, skin conditions, scans, and other medical images"
)

if uploaded_image:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_image, caption="📎 Image ready for analysis", use_column_width=True)

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "",
        placeholder="Ask MediChat a medical question... or describe the uploaded image",
        label_visibility="collapsed"
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.form_submit_button("Send Message 💬")

st.markdown('</div>', unsafe_allow_html=True)

# ── Handle Input ──────────────────────────────────────────────────────
if submit and (user_input.strip() or uploaded_image):
    st.session_state.question_count += 1

    if uploaded_image:
        st.session_state.messages.append({
            "role": "user",
            "type": "image",
            "content": user_input.strip()
        })
        with st.spinner("🔍 MediChat is analysing the image..."):
            uploaded_image.seek(0)
            image_b64 = encode_image(uploaded_image)
            reply = medichat_vision(user_input, image_b64)
    else:
        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": user_input.strip()
        })
        with st.spinner("🔬 MediChat is searching medical literature..."):
            reply = medichat_rag(user_input, st.session_state.messages[:-1])

    st.session_state.messages.append({
        "role": "assistant",
        "type": "text",
        "content": reply
    })
    st.rerun()
