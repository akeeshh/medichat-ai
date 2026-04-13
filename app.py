
import streamlit as st
from groq import Groq

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediChat — Clinical AI Assistant",
    page_icon="🏥",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0d2b55 50%, #0b3d6b 100%);
        min-height: 100vh;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c9a7, #0b7b8b, #00c9a7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.3rem;
        font-weight: 300;
    }

    .chat-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
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

    .user-name {
        color: #ffffff99;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
        text-align: right;
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
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 201, 167, 0.35) !important;
    }

    .stats-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 0.5rem 0 1.5rem 0;
    }

    .stat-item {
        text-align: center;
    }

    .stat-num {
        color: #00c9a7;
        font-size: 1.2rem;
        font-weight: 700;
    }

    .stat-label {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 500;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }

    .welcome-msg {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        padding: 2rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ── API Setup ────────────────────────────────────────────────────────
GROQ_API_KEY = "GROQ_API_KEY"
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are MediChat, a professional and empathetic clinical AI assistant designed to help patients and healthcare professionals with medical information.

Your role:
- Provide clear, accurate, and evidence-based medical information
- Explain medical terms in simple, understandable language  
- Be empathetic and supportive in your responses
- Always recommend consulting a qualified doctor for personal medical decisions
- Never diagnose conditions — only provide general medical information
- Structure responses clearly using bullet points or numbered lists when helpful

Always end responses with a gentle reminder to consult a healthcare professional for personal medical advice."""

# ── Session State ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">🏥 MediChat</div>
    <div class="main-subtitle">Clinical AI Assistant — Powered by Advanced Language Models</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-num">{st.session_state.question_count}</div>
        <div class="stat-label">QUESTIONS ASKED</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">AI</div>
        <div class="stat-label">POWERED</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">24/7</div>
        <div class="stat-label">AVAILABLE</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ MediChat provides general medical information only. Always consult a qualified healthcare professional for personal medical advice.
</div>
""", unsafe_allow_html=True)

# ── Chat Display ─────────────────────────────────────────────────────
chat_area = st.container()
with chat_area:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-msg">
            👋 Welcome to MediChat! Ask me any medical question and I'll do my best to help.<br>
            <br>Try asking: "What are the symptoms of hypertension?" or "How does the immune system work?"
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-name">You</div>
                <div class="user-bubble">{msg["content"]}</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-name">🏥 MediChat</div>
                <div class="bot-bubble">{msg["content"]}</div>
                """, unsafe_allow_html=True)

# ── Input Area ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "",
        placeholder="Ask MediChat a medical question...",
        label_visibility="collapsed"
    )
    submit = st.form_submit_button("Send Message 💬")

# ── Handle Input ─────────────────────────────────────────────────────
if submit and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.question_count += 1

    with st.spinner("MediChat is thinking..."):
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        api_messages += [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# ── Clear Button ─────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()

