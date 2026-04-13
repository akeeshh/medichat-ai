import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import os

st.set_page_config(
    page_title="MediChat — Clinical AI Assistant",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0d2b55 50%, #0b3d6b 100%);
        min-height: 100vh;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c9a7, #0b7b8b, #00c9a7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        text-align: center;
        margin-top: 0.3rem;
        font-weight: 300;
    }
    .user-bubble {
        background: linear-gradient(135deg, #0b7b8b, #00897b);
        color: white;
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        margin-left: 15%;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(0, 201, 167, 0.2);
    }
    .bot-bubble {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        margin-right: 5%;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .bot-name {
        color: #00c9a7;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .disclaimer {
        background: rgba(251, 191, 36, 0.08);
        border: 1px solid rgba(251, 191, 36, 0.2);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        color: #fbbf24;
        font-size: 0.8rem;
        margin: 0.5rem 0 1rem 0;
        text-align: center;
    }
    .rag-badge {
        background: rgba(0, 201, 167, 0.1);
        border: 1px solid rgba(0, 201, 167, 0.3);
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        color: #00c9a7;
        font-size: 0.75rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 0.5rem 0 1.5rem 0;
    }
    .stat-item { text-align: center; }
    .stat-num { color: #00c9a7; font-size: 1.2rem; font-weight: 700; }
    .stat-label { color: #64748b; font-size: 0.7rem; font-weight: 500; }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00c9a7 !important;
        box-shadow: 0 0 0 2px rgba(0, 201, 167, 0.2) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00c9a7, #0b7b8b) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
    }
    .welcome-msg {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        padding: 2rem;
        font-style: italic;
    }
    div[data-testid="stMarkdownContainer"] p { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Load API Key ─────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("API key not found. Please check your secrets configuration.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Load RAG Components (cached so they only load once) ───────────────
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

with st.spinner("Loading MediChat knowledge base... first load takes ~30 seconds"):
    embedder, index, documents = load_rag_system()

# ── RAG Function ─────────────────────────────────────────────────────
def medichat_rag(user_question, chat_history):
    question_embedding = embedder.encode([user_question]).astype('float32')
    distances, indices = index.search(question_embedding, k=3)
    relevant_docs = [documents[i] for i in indices[0]]
    context = "\n\n---\n\n".join(relevant_docs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are MediChat, a professional and empathetic clinical AI assistant. "
                "Use the following real medical research context to answer accurately. "
                "Structure your answers clearly with bullet points where helpful. "
                "Always recommend consulting a qualified doctor for personal medical advice.\n\n"
                "MEDICAL RESEARCH CONTEXT:\n" + context
            )
        }
    ]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_question})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# ── Session State ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# ── Header ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 MediChat</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Clinical AI Assistant — Powered by Advanced Language Models</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-num">{st.session_state.question_count}</div>
        <div class="stat-label">QUESTIONS ASKED</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">300</div>
        <div class="stat-label">PUBMED DOCS</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">24/7</div>
        <div class="stat-label">AVAILABLE</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="rag-badge">🔬 RAG Pipeline Active — Answers grounded in real PubMed research</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ MediChat provides general medical information only. Always consult a qualified healthcare professional for personal medical advice.
</div>
""", unsafe_allow_html=True)

# ── Chat Display ──────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-msg">
        👋 Welcome to MediChat! Ask me any medical question.<br><br>
        Try: "What causes high blood pressure?" or "How does diabetes affect the body?"
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-name">🏥 MediChat</div><div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="Ask MediChat a medical question...", label_visibility="collapsed")
    submit = st.form_submit_button("Send Message 💬")

if submit and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.question_count += 1
    with st.spinner("MediChat is searching medical literature..."):
        reply = medichat_rag(user_input, st.session_state.messages[:-1])
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

if st.session_state.messages:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()
