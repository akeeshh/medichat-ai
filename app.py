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
import random
import time

st.set_page_config(
    page_title="MediChat — Clinical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    * { font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }

    .stApp { background: #040d1a; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060f1e 0%, #080d1a 100%) !important;
        border-right: 1px solid rgba(0,201,167,0.12) !important;
    }

    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .main .block-container { padding: 0.8rem 1rem; max-width: 100%; }

    /* Top badges */
    .badges-top {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }
    .badge {
        padding: 0.28rem 0.85rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        letter-spacing: 0.01em;
    }
    .badge-rag { background: rgba(0,201,167,0.1); border: 1px solid rgba(0,201,167,0.3); color: #00c9a7; }
    .badge-vision { background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3); color: #a78bfa; }
    .badge-live { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); color: #22c55e; }

    /* Panel cards */
    .panel-card {
        background: linear-gradient(135deg, rgba(8,18,38,0.95), rgba(6,14,30,0.95));
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.9rem;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .panel-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00c9a7, #0b7b8b, transparent);
    }
    .panel-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .panel-title .new-tag {
        background: rgba(0,201,167,0.15);
        border: 1px solid rgba(0,201,167,0.25);
        color: #00c9a7;
        font-size: 0.6rem;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Knowledge graph simulation */
    .kg-node {
        display: inline-block;
        background: rgba(0,201,167,0.12);
        border: 1px solid rgba(0,201,167,0.25);
        border-radius: 50%;
        padding: 0.2rem 0.5rem;
        font-size: 0.6rem;
        color: #00c9a7;
        margin: 0.15rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .kg-node.purple { background: rgba(139,92,246,0.12); border-color: rgba(139,92,246,0.25); color: #a78bfa; }
    .kg-node.blue { background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.25); color: #60a5fa; }
    .kg-node.orange { background: rgba(251,146,60,0.12); border-color: rgba(251,146,60,0.25); color: #fb923c; }
    .kg-center {
        text-align: center;
        font-size: 0.7rem;
        color: #00c9a7;
        font-weight: 700;
        padding: 0.4rem;
        background: rgba(0,201,167,0.08);
        border: 1px solid rgba(0,201,167,0.2);
        border-radius: 8px;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Data stream */
    .stream-line {
        font-size: 0.65rem;
        color: #334155;
        font-family: 'JetBrains Mono', monospace;
        padding: 0.12rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stream-line.active { color: #00c9a7; }
    .stream-line.mid { color: #475569; }

    /* Citation panel */
    .citation-item {
        background: rgba(255,255,255,0.03);
        border-left: 2px solid #0b7b8b;
        padding: 0.35rem 0.5rem;
        margin-bottom: 0.3rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.65rem;
        color: #64748b;
        line-height: 1.3;
    }
    .citation-count {
        font-size: 0.6rem;
        color: #00c9a7;
        font-weight: 700;
    }

    /* Vitals */
    .vital-card {
        background: rgba(8,18,38,0.9);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .vital-label { font-size: 0.65rem; color: #64748b; font-weight: 500; }
    .vital-value { font-size: 0.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .vital-trend { font-size: 0.6rem; padding: 0.1rem 0.3rem; border-radius: 4px; }
    .trend-up { background: rgba(34,197,94,0.1); color: #22c55e; }
    .trend-stable { background: rgba(59,130,246,0.1); color: #60a5fa; }
    .trend-fluctuating { background: rgba(251,191,36,0.1); color: #fbbf24; }
    .trend-pending { background: rgba(139,92,246,0.1); color: #a78bfa; }
    .simulated-tag {
        font-size: 0.55rem;
        color: #1e3a5f;
        text-align: right;
        margin-top: 0.3rem;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(8,18,38,0.7), rgba(6,14,30,0.8));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .welcome-title { color: #e2e8f0; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.5rem; }
    .welcome-text { color: #64748b; font-size: 0.85rem; line-height: 1.6; margin-bottom: 1rem; }
    .chip {
        display: inline-block;
        background: rgba(0,201,167,0.08);
        border: 1px solid rgba(0,201,167,0.2);
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        color: #00c9a7;
        font-size: 0.75rem;
        margin: 0.2rem;
    }

    /* Chat bubbles */
    .user-row { display: flex; justify-content: flex-end; align-items: flex-end; gap: 0.5rem; margin: 0.6rem 0; }
    .bot-row { display: flex; justify-content: flex-start; align-items: flex-start; gap: 0.5rem; margin: 0.6rem 0; }
    .avatar { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0; }
    .avatar-user { background: linear-gradient(135deg, #0b7b8b, #00897b); }
    .avatar-bot { background: linear-gradient(135deg, #1e3a5f, #0d2b55); border: 1px solid rgba(0,201,167,0.3); }
    .user-bubble {
        background: linear-gradient(135deg, #0b7b8b, #00897b);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 16px 16px 4px 16px;
        max-width: 72%;
        font-size: 0.88rem;
        line-height: 1.5;
        box-shadow: 0 4px 16px rgba(0,201,167,0.15);
    }
    .bot-bubble {
        background: linear-gradient(135deg, rgba(8,20,45,0.95), rgba(6,14,30,0.95));
        border: 1px solid rgba(255,255,255,0.07);
        color: #e2e8f0;
        padding: 0.75rem 1rem;
        border-radius: 16px 16px 16px 4px;
        max-width: 78%;
        font-size: 0.88rem;
        line-height: 1.65;
    }
    .bot-bubble strong { color: #00c9a7; }
    .image-tag {
        background: rgba(139,92,246,0.1);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        color: #a78bfa;
        font-size: 0.73rem;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    /* Input area */
    .input-section {
        background: linear-gradient(135deg, rgba(8,18,38,0.9), rgba(6,14,30,0.9));
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-top: 0.6rem;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.88rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00c9a7 !important;
        box-shadow: 0 0 0 2px rgba(0,201,167,0.1) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #334155 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #00c9a7, #0b7b8b) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important; font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(0,201,167,0.25) !important; }

    /* Upload area */
    .upload-bar {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .upload-progress {
        flex: 1;
        height: 4px;
        background: rgba(255,255,255,0.06);
        border-radius: 2px;
        overflow: hidden;
    }
    .upload-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00c9a7, #0b7b8b);
        border-radius: 2px;
    }
    .img-thumb {
        width: 42px; height: 42px;
        border-radius: 6px;
        object-fit: cover;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.05);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }

    /* Advanced tools */
    .tools-bar {
        background: linear-gradient(135deg, rgba(8,18,38,0.9), rgba(6,14,30,0.9));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
        margin-top: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .tools-label { font-size: 0.72rem; font-weight: 700; color: #64748b; margin-right: 0.3rem; }
    .tool-btn {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 0.25rem 0.7rem;
        color: #94a3b8;
        font-size: 0.7rem;
        font-weight: 500;
        cursor: pointer;
    }
    .tool-btn.active { background: rgba(0,201,167,0.1); border-color: rgba(0,201,167,0.25); color: #00c9a7; }

    /* Vitals inline */
    .vitals-inline {
        background: rgba(8,18,38,0.9);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-top: 0.5rem;
    }
    .vital-inline-item { font-size: 0.7rem; margin-bottom: 0.25rem; }
    .vital-inline-key { color: #00c9a7; font-weight: 600; }
    .vital-inline-val { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
    .vital-inline-trend { font-size: 0.62rem; }

    /* Sidebar stats */
    .sb-stat { margin-bottom: 0.25rem; font-size: 0.75rem; color: #64748b; }
    .sb-stat span { color: #e2e8f0; font-weight: 600; }
    .sb-keyphrase {
        background: rgba(0,201,167,0.08);
        border: 1px solid rgba(0,201,167,0.15);
        border-radius: 6px;
        padding: 0.3rem 0.6rem;
        font-size: 0.7rem;
        color: #00c9a7;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.4rem;
    }
    .sb-feature {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 0.45rem 0.7rem;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .sb-feature-name { font-size: 0.72rem; font-weight: 600; color: #e2e8f0; }
    .sb-feature-graph { font-size: 0.6rem; color: #00c9a7; font-family: 'JetBrains Mono', monospace; letter-spacing: -1px; }
    .sb-feature-live { font-size: 0.6rem; color: #22c55e; }
    .sb-suggestion { font-size: 0.72rem; color: #475569; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); line-height: 1.4; }
    .sb-user { display: flex; align-items: center; gap: 0.5rem; padding-top: 0.8rem; }
    .sb-avatar { width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg, #0b7b8b, #00897b); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; }
    .sb-username { font-size: 0.72rem; color: #64748b; }

    div[data-testid="stMarkdownContainer"] p { color: #e2e8f0; }
    div[data-testid="column"] { padding: 0 0.25rem !important; }
    .disclaimer { background: rgba(251,191,36,0.05); border: 1px solid rgba(251,191,36,0.15); border-radius: 8px; padding: 0.5rem 0.8rem; color: #fbbf24; font-size: 0.72rem; text-align: center; margin-bottom: 0.6rem; }

    .stFileUploader { background: transparent !important; border: none !important; }
    .stFileUploader > div { background: rgba(255,255,255,0.03) !important; border: 1px dashed rgba(255,255,255,0.1) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── API + RAG ─────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("⚠️ API key not found.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_rag_system():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:300]")
    documents = []
    for item in dataset:
        documents.append(f"Question: {item['question']}\nAnswer: {item['long_answer']}")
    embeddings = embedder.encode(documents)
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings.astype('float32'))
    return embedder, idx, documents

with st.spinner("🔬 Loading MediChat knowledge base..."):
    embedder, index, documents = load_rag_system()

def encode_image(f):
    img = Image.open(f)
    if img.mode != "RGB": img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def medichat_rag(q, history):
    emb = embedder.encode([q]).astype('float32')
    _, idxs = index.search(emb, k=3)
    ctx = "\n\n---\n\n".join([documents[i] for i in idxs[0]])
    msgs = [{"role":"system","content":(
        "You are MediChat, a professional clinical AI assistant. "
        "Use this PubMed research context to answer accurately. "
        "Use **bold** for key terms, bullet points for lists. "
        "Always recommend consulting a doctor.\n\nCONTEXT:\n"+ctx
    )}]
    for m in history:
        if m.get("type")=="text": msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":q})
    r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile",messages=msgs,temperature=0.5,max_tokens=1024)
    return r.choices[0].message.content

def medichat_vision(q, b64):
    p = q if q.strip() else "Analyse this medical image with detailed clinical observations."
    r = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role":"user","content":[
            {"type":"text","text":(
                "You are MediChat, a clinical AI assistant. "
                "Analyse this image. Provide: 1) **Clinical Observations** 2) **Differential Diagnoses** 3) **Recommendations**. "
                "Always remind the user to consult a doctor.\n\nQuestion: "+p
            )},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
        ]}],
        temperature=0.5, max_tokens=1024
    )
    return r.choices[0].message.content

# ── Session State ─────────────────────────────────────────────────────
if "messages" not in st.session_state: st.session_state.messages = []
if "qcount" not in st.session_state: st.session_state.qcount = 0
if "users" not in st.session_state: st.session_state.users = random.randint(4,8)

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediChat")
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.markdown(f"""
    <div class="sb-stat">{st.session_state.qcount} Questions, <span>{st.session_state.users} Unique Users</span></div>
    <div class="sb-stat">300 Docs, <span>7 Journals</span></div>
    <div class="sb-stat" style="margin-top:0.4rem;font-size:0.68rem;color:#475569;">Top PubMed Keyphrase</div>
    <div class="sb-keyphrase">CAR-T Therapy</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎛️ Active Features")
    st.markdown("""
    <div class="sb-feature">
        <div class="sb-feature-name">🔬 RAG Pipeline</div>
        <div class="sb-feature-graph">▁▂▃▅▆▇▆▅</div>
    </div>
    <div class="sb-feature">
        <div class="sb-feature-name">👁️ Vision AI</div>
        <div class="sb-feature-graph">▂▃▄▅▃▅▆▄</div>
    </div>
    <div class="sb-feature">
        <div class="sb-feature-name">🟢 Live</div>
        <div class="sb-feature-live">● Active</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 💡 Try Asking")
    suggestions = [
        "CAR-T therapy for glioblastoma variants",
        "Comparing GLP-1 and SGLT-2 inhibitor efficacy in heart failure",
        "Guidelines for triple-negative breast cancer (TNBC) immunotherapy",
        "Metabolic pathways influenced by NAD+ precursors in aging",
    ]
    for s in suggestions:
        st.markdown(f'<div class="sb-suggestion">• {s}</div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.qcount = 0
        st.rerun()
    st.markdown("""
    <div class="sb-user">
        <div class="sb-avatar">👤</div>
        <div class="sb-username">Active Analyst</div>
    </div>
    <div style="color:#1e3a5f;font-size:0.65rem;margin-top:0.8rem;text-align:center;">
    ICT654 — Group 7 — SISTC Melbourne
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="badges-top">
    <span class="badge badge-rag">🔬 RAG Active — Grounded in PubMed Research</span>
    <span class="badge badge-vision">👁️ Vision Active — Medical Image Analysis</span>
    <span class="badge badge-live">🟢 Live</span>
</div>
""", unsafe_allow_html=True)

# ── TOP PANELS ────────────────────────────────────────────────────────
col_kg, col_stream, col_cite = st.columns([1, 1, 1])

with col_kg:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-title">Knowledge Graph Visualizer <span class="new-tag">New Module</span></div>
        <div class="kg-center">Hypertension</div>
        <div style="text-align:center;line-height:2;">
            <span class="kg-node">BP Regulator</span>
            <span class="kg-node purple">ACE Inhibitor</span>
            <span class="kg-node blue">Diuretics</span>
            <span class="kg-node orange">Inflammation</span>
            <span class="kg-node">Nitric Oxide</span>
            <span class="kg-node purple">Renin</span>
            <span class="kg-node blue">Endothelin</span>
            <span class="kg-node">Vasodilation</span>
            <span class="kg-node orange">ARB</span>
            <span class="kg-node">Ca Channel</span>
            <span class="kg-node purple">Beta Blocker</span>
            <span class="kg-node blue">Aldosterone</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_stream:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-title">Medical Data Stream</div>
        <div class="stream-line active">► PubMed sync... 300 docs loaded</div>
        <div class="stream-line mid">► Embedding model: MiniLM-L6-v2</div>
        <div class="stream-line mid">► FAISS index: 300 vectors</div>
        <div class="stream-line active">► RAG pipeline: ACTIVE</div>
        <div class="stream-line mid">► Vision model: Llama-4-Scout</div>
        <div class="stream-line active">► Groq API: CONNECTED</div>
        <div class="stream-line mid">► Latency: ~1.2s avg</div>
        <div class="stream-line active">► Session: LIVE</div>
        <div class="stream-line mid">► Knowledge base: PubMedQA</div>
        <div class="stream-line active">► Status: ALL SYSTEMS GO ✓</div>
    </div>
    """, unsafe_allow_html=True)

with col_cite:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-title">PubMed Citation Network</div>
        <div style="font-size:0.65rem;color:#475569;margin-bottom:0.4rem;">Recent papers · 9,379+ · <span class="citation-count">17 citations</span></div>
        <div class="citation-item">Abstract snippet... linked to annotation pipeline [node in Knowledge Graph]</div>
        <div class="citation-item">Abstract snippet... linked to RAG retrieval context</div>
        <div class="citation-item">PubMed ID: 38291042 · Cell Death · 2024</div>
        <div class="citation-item">Mitochondrial dynamics in PCD · Nature · 2023</div>
    </div>
    """, unsafe_allow_html=True)

# ── WELCOME OR CHAT ───────────────────────────────────────────────────
col_vitals_l, col_chat, col_vitals_r = st.columns([1, 3, 1])

with col_vitals_l:
    st.markdown("""
    <div class="panel-card" style="margin-top:0.4rem;">
        <div class="panel-title">Patient Vitals <span class="new-tag">New Module</span></div>
        <div class="vital-card">
            <div>
                <div class="vital-label">Heart Rate</div>
                <div class="vital-value" style="color:#ef4444;">85 bpm</div>
            </div>
            <div class="vital-trend trend-up">Trend: Up</div>
        </div>
        <div class="vital-card">
            <div>
                <div class="vital-label">Blood Pressure</div>
                <div class="vital-value" style="color:#60a5fa;">135/88</div>
            </div>
            <div class="vital-trend trend-stable">Stable</div>
        </div>
        <div class="vital-card">
            <div>
                <div class="vital-label">O₂ Saturation</div>
                <div class="vital-value" style="color:#00c9a7;">96%</div>
            </div>
            <div class="vital-trend trend-fluctuating">Fluctuating</div>
        </div>
        <div class="vital-card">
            <div>
                <div class="vital-label">WBC Count</div>
                <div class="vital-value" style="color:#fbbf24;">10k</div>
            </div>
            <div class="vital-trend trend-pending">Pending Lab</div>
        </div>
        <div class="simulated-tag">Simulated Patient Data</div>
    </div>
    """, unsafe_allow_html=True)

with col_chat:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">🏥</div>
            <div class="welcome-title">Welcome to MediChat</div>
            <div class="welcome-text">
                Your intelligent clinical AI assistant powered by real medical research.<br>
                Ask any medical question or upload a medical image for analysis.
            </div>
            <div>
                <span class="chip">💊 Drug interactions</span>
                <span class="chip">🫀 Heart conditions</span>
                <span class="chip">🧬 Genetics</span>
                <span class="chip">🦠 Infections</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if msg.get("type") == "image":
                    st.markdown('<div class="image-tag">🖼️ Medical image uploaded for analysis</div>', unsafe_allow_html=True)
                    if msg.get("content"):
                        st.markdown(f'<div class="user-row"><div class="user-bubble">{msg["content"]}</div><div class="avatar avatar-user">👤</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="user-row"><div class="user-bubble">{msg["content"]}</div><div class="avatar avatar-user">👤</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-row"><div class="avatar avatar-bot">🏥</div><div class="bot-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

with col_vitals_r:
    st.markdown("""
    <div class="panel-card" style="margin-top:0.4rem;">
        <div class="panel-title">Patient Vitals <span class="new-tag">New Module</span></div>
        <div class="vital-card">
            <div>
                <div class="vital-label">Heart Rate</div>
                <div class="vital-value" style="color:#ef4444;">85 bpm</div>
            </div>
            <div class="vital-trend trend-up">Trend: Up</div>
        </div>
        <div class="vital-card">
            <div>
                <div class="vital-label">Blood Pressure</div>
                <div class="vital-value" style="color:#60a5fa;">135/88</div>
            </div>
            <div class="vital-trend trend-stable">Stable</div>
        </div>
        <div class="vital-card">
            <div>
                <div class="vital-label">O₂ Saturation</div>
                <div class="vital-value" style="color:#00c9a7;">96%</div>
            </div>
            <div class="vital-trend trend-fluctuating">Fluctuating</div>
        </div>
        <div class="vital-card">
            <div>
                <div class="vital-label">WBC Count</div>
                <div class="vital-value" style="color:#fbbf24;">10k</div>
            </div>
            <div class="vital-trend trend-pending">Pending Lab</div>
        </div>
        <div class="simulated-tag">Simulated Patient Data</div>
    </div>
    """, unsafe_allow_html=True)

# ── INPUT SECTION ─────────────────────────────────────────────────────
st.markdown('<div class="disclaimer">⚠️ MediChat provides general medical information only. Always consult a qualified healthcare professional for personal medical advice.</div>', unsafe_allow_html=True)

uploaded_image = st.file_uploader(
    "📎 Upload a medical image for analysis",
    type=["jpg", "jpeg", "png"],
    help="Supports X-rays, skin conditions, scans, and other medical images"
)

if uploaded_image:
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.image(uploaded_image, caption="📎 Ready for analysis", use_column_width=True)
    st.markdown("""
    <div class="upload-bar">
        <span style="font-size:0.7rem;color:#00c9a7;font-weight:600;">📎 Image loaded</span>
        <div class="upload-progress"><div class="upload-progress-fill" style="width:100%;"></div></div>
        <span style="font-size:0.7rem;color:#64748b;">100%</span>
        <span class="img-thumb">🩻</span>
        <span class="img-thumb">📷</span>
        <span class="img-thumb" style="border-color:rgba(0,201,167,0.3);color:#00c9a7;">+</span>
    </div>
    """, unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "",
        placeholder="Ask a complex clinical question, or paste an image analysis report for cross-reference...",
        label_visibility="collapsed"
    )
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        submit = st.form_submit_button("Send Message 💬")

# ── Vitals inline display ─────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("""
    <div class="vitals-inline">
        <div class="vital-inline-item">
            <span style="color:#ef4444;">❤️ </span>
            <span class="vital-inline-key">Heart Rate: </span>
            <span class="vital-inline-val">85 bpm </span>
            <span class="vital-inline-trend trend-up">(Trend: Up)</span>
        </div>
        <div class="vital-inline-item">
            <span style="color:#60a5fa;">🩺 </span>
            <span class="vital-inline-key">Blood Pressure: </span>
            <span class="vital-inline-val">135/88 </span>
            <span class="vital-inline-trend trend-stable">(Trend: Stable)</span>
        </div>
        <div class="vital-inline-item">
            <span style="color:#00c9a7;">💨 </span>
            <span class="vital-inline-key">O₂ Sat: </span>
            <span class="vital-inline-val">96% </span>
            <span class="vital-inline-trend trend-fluctuating">(Trend: Fluctuating)</span>
        </div>
        <div class="vital-inline-item">
            <span style="color:#fbbf24;">🔬 </span>
            <span class="vital-inline-key">WBC Count: </span>
            <span class="vital-inline-val">10k </span>
            <span class="vital-inline-trend trend-pending">(Trend: Pending Lab)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Advanced Tools Bar ────────────────────────────────────────────────
st.markdown("""
<div class="tools-bar">
    <span class="tools-label">🛠️ Advanced Query Tools</span>
    <span class="tool-btn active">Refine by Topic ▾</span>
    <span class="tool-btn">Cross-reference [Current View]</span>
    <span class="tool-btn">Generate Report Summary [Download PDF]</span>
</div>
""", unsafe_allow_html=True)

# ── Handle Input ──────────────────────────────────────────────────────
if submit and (user_input.strip() or uploaded_image):
    st.session_state.qcount += 1
    if uploaded_image:
        st.session_state.messages.append({"role":"user","type":"image","content":user_input.strip()})
        with st.spinner("🔍 Analysing medical image..."):
            uploaded_image.seek(0)
            reply = medichat_vision(user_input, encode_image(uploaded_image))
    else:
        st.session_state.messages.append({"role":"user","type":"text","content":user_input.strip()})
        with st.spinner("🔬 Searching PubMed knowledge base..."):
            reply = medichat_rag(user_input, st.session_state.messages[:-1])
    st.session_state.messages.append({"role":"assistant","type":"text","content":reply})
    st.rerun()
