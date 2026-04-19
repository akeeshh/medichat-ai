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
from fpdf import FPDF

st.set_page_config(
    page_title="MediChat - Your Health Assistant",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(160deg, #f0f9ff 0%, #e0f2fe 40%, #f0fdf4 100%); min-height: 100vh; }
    .main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 820px; }
    .header-card { background: white; border-radius: 20px; padding: 1.4rem 2rem; margin-bottom: 1rem; box-shadow: 0 4px 24px rgba(0,0,0,0.06); display: flex; align-items: center; gap: 1.2rem; border: 1px solid rgba(255,255,255,0.8); }
    .header-title { font-size: 1.8rem; font-weight: 800; color: #0f766e; margin: 0; }
    .header-subtitle { color: #64748b; font-size: 0.85rem; margin: 0.2rem 0 0 0; }
    .stats-row { display: flex; gap: 0.6rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
    .stat-pill { background: white; border: 1px solid #e2e8f0; border-radius: 50px; padding: 0.3rem 0.8rem; font-size: 0.72rem; font-weight: 600; color: #475569; }
    .stat-pill.green { color: #0f766e; border-color: #99f6e4; background: #f0fdfa; }
    .stat-pill.blue { color: #0369a1; border-color: #bae6fd; background: #f0f9ff; }
    .stat-pill.purple { color: #7c3aed; border-color: #ddd6fe; background: #faf5ff; }
    .stat-pill.orange { color: #c2410c; border-color: #fed7aa; background: #fff7ed; }
    .disclaimer { background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px; padding: 0.6rem 1rem; color: #92400e; font-size: 0.76rem; margin-bottom: 0.8rem; text-align: center; }
    .welcome-card { background: white; border-radius: 20px; padding: 2.2rem 2rem; text-align: center; box-shadow: 0 4px 24px rgba(0,0,0,0.06); margin: 0.5rem 0 1rem 0; }
    .welcome-title { font-size: 1.3rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem; }
    .welcome-text { color: #64748b; font-size: 0.87rem; line-height: 1.6; margin-bottom: 1rem; }
    .chip-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.4rem; }
    .chip { background: #f0fdf4; border: 1px solid #86efac; border-radius: 50px; padding: 0.3rem 0.8rem; color: #166534; font-size: 0.76rem; font-weight: 500; }
    .assessment-card { background: white; border-radius: 20px; padding: 1.5rem 1.8rem; box-shadow: 0 4px 24px rgba(0,0,0,0.06); margin-bottom: 1rem; border: 1px solid #e2e8f0; }
    .assessment-title { font-size: 1rem; font-weight: 700; color: #0f766e; margin-bottom: 0.3rem; }
    .assessment-subtitle { font-size: 0.78rem; color: #64748b; margin-bottom: 1rem; }
    .progress-bar-wrap { background: #f1f5f9; border-radius: 50px; height: 6px; margin-bottom: 1.2rem; overflow: hidden; }
    .progress-bar-fill { height: 100%; background: linear-gradient(90deg, #0d9488, #059669); border-radius: 50px; }
    .progress-label { font-size: 0.7rem; color: #94a3b8; margin-bottom: 0.3rem; display: flex; justify-content: space-between; }
    .question-bubble { background: linear-gradient(135deg, #f0fdfa, #ecfdf5); border: 1px solid #99f6e4; border-radius: 16px 16px 16px 4px; padding: 1rem 1.2rem; margin-bottom: 1rem; font-size: 0.95rem; color: #134e4a; font-weight: 500; line-height: 1.5; }
    .user-wrap { display: flex; justify-content: flex-end; align-items: flex-end; gap: 0.5rem; margin: 0.6rem 0; }
    .bot-wrap { display: flex; justify-content: flex-start; align-items: flex-start; gap: 0.5rem; margin: 0.6rem 0; }
    .av { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; flex-shrink: 0; color: white; }
    .av-user { background: linear-gradient(135deg, #0d9488, #059669); }
    .av-bot { background: #0f766e; border: 2px solid #99f6e4; }
    .user-bubble { background: linear-gradient(135deg, #0d9488, #059669); color: white; padding: 0.75rem 1rem; border-radius: 16px 16px 4px 16px; max-width: 75%; font-size: 0.9rem; line-height: 1.5; }
    .bot-bubble { background: white; border: 1px solid #e2e8f0; color: #1e293b; padding: 0.75rem 1rem; border-radius: 16px 16px 16px 4px; max-width: 78%; font-size: 0.9rem; line-height: 1.65; }
    .bot-bubble strong { color: #0f766e; }
    .bot-label { font-size: 0.7rem; font-weight: 700; color: #0f766e; margin-bottom: 0.25rem; margin-left: 40px; }
    .image-tag { background: #faf5ff; border: 1px solid #ddd6fe; border-radius: 10px; padding: 0.3rem 0.7rem; color: #7c3aed; font-size: 0.73rem; margin-bottom: 0.3rem; display: inline-block; }
    .memory-card { background: #f0fdf4; border: 1px solid #86efac; border-radius: 12px; padding: 0.6rem 1rem; margin-bottom: 0.8rem; font-size: 0.76rem; color: #166534; }
    .memory-title { font-weight: 700; margin-bottom: 0.25rem; font-size: 0.78rem; }

    .emergency-banner {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        color: white;
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 20px rgba(220,38,38,0.35);
        animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 6px 20px rgba(220,38,38,0.35); }
        50% { box-shadow: 0 6px 30px rgba(220,38,38,0.65); }
    }
    .emergency-title { font-size: 1.1rem; font-weight: 800; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.5rem; }
    .emergency-text { font-size: 0.85rem; line-height: 1.5; margin-bottom: 0.5rem; }
    .emergency-number {
        background: white; color: #991b1b;
        padding: 0.5rem 1rem; border-radius: 10px;
        font-size: 1.1rem; font-weight: 800;
        display: inline-block; margin-top: 0.2rem;
        letter-spacing: 0.05em;
    }

    .source-tag {
        display: inline-block;
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        color: #0369a1;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 50px;
        margin-right: 0.3rem;
        margin-top: 0.3rem;
    }
    .source-row {
        margin-left: 42px;
        margin-bottom: 0.2rem;
        font-size: 0.7rem;
        color: #64748b;
    }

    .name-welcome {
        background: linear-gradient(135deg, #f0fdfa, #ecfdf5);
        border: 1px solid #99f6e4;
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .name-welcome-text {
        font-size: 0.92rem;
        color: #134e4a;
        font-weight: 500;
    }
    .input-card { background: white; border-radius: 20px; padding: 1.1rem 1.3rem; box-shadow: 0 4px 24px rgba(0,0,0,0.06); margin-top: 0.8rem; border: 1px solid #e2e8f0; }
    .stTextInput > div > div > input { background: #f8fafc !important; border: 1.5px solid #e2e8f0 !important; border-radius: 12px !important; color: #1e293b !important; padding: 0.75rem 1rem !important; font-size: 0.9rem !important; }
    .stTextInput > div > div > input:focus { border-color: #0d9488 !important; box-shadow: 0 0 0 3px rgba(13,148,136,0.1) !important; }
    .stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }
    .stButton > button { background: linear-gradient(135deg, #0d9488, #059669) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.65rem 1.3rem !important; font-weight: 600 !important; font-size: 0.88rem !important; }
    .stFileUploader > div { background: #f8fafc !important; border: 1.5px dashed #cbd5e1 !important; border-radius: 12px !important; }
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

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("API key not found.")
    st.stop()
groq_client = Groq(api_key=GROQ_API_KEY)

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
    pubmed = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:500]")
    pubmed_docs = ["[PubMed Research]\nQuestion: " + i["question"] + "\nAnswer: " + i["long_answer"] for i in pubmed]
    meddialog = load_dataset("BinKhoaLe1812/MedDialog-EN-100k", split="train[:500]")
    dialog_docs = ["[Doctor-Patient Conversation]\nPatient: " + i["input"] + "\nDoctor: " + i["output"] for i in meddialog]
    docs = pubmed_docs + dialog_docs
    embeddings = embedder.encode(docs)
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings.astype("float32"))
    return embedder, idx, docs

with st.spinner("Loading MediChat knowledge base..."):
    embedder, index, documents = load_rag_system()

def encode_image(f):
    img = Image.open(f)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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

# ── Emergency Detection ───────────────────────────────────────────────
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

def detect_emergency(text):
    if not text:
        return False
    text_lower = text.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in text_lower:
            return True
    return False

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
    _, idxs = index.search(emb, k=3)
    context = "\n\n---\n\n".join([documents[i] for i in idxs[0]])
    sources = get_sources_used(idxs[0])
    memory = extract_patient_memory(all_messages)
    memory_context = build_memory_context(memory)
    history = []
    for m in all_messages[-10:]:
        if m.get("type") == "text":
            history.append({"role": m["role"], "content": m["content"]})
    system = (
        "You are MediChat, a confident and clinically knowledgeable AI health assistant. "
        "Your patients come to you because their doctors have not given them clear answers. "
        "Your job is to be genuinely useful, not to over-disclaim or be evasive.\n\n"
        "HOW TO RESPOND:\n"
        "1. Be direct and concrete. Patients want real answers, not endless validation.\n"
        "2. When a patient describes clear symptom clusters (e.g. racing heart + nausea + shortness of breath + feeling faint), name the likely conditions plainly. Do not hide behind 'I'm not a doctor' every sentence.\n"
        "3. Use the medical context below to identify the most likely diagnoses and explain them in simple language.\n"
        "4. Only include ONE disclaimer at the END of your response, not in every paragraph. The app already shows a disclaimer banner.\n"
        "5. Keep responses focused: name the likely condition(s), explain briefly, give concrete next steps.\n"
        "6. Don't repeat the patient's symptoms back to them in every message — they already know.\n"
        "7. Be warm but confident. You help them MORE by giving real information than by saying 'that sounds scary'.\n"
        "8. If symptoms match a well-known pattern (panic attack, migraine, asthma, hypoglycemia, vertigo, etc.), SAY SO.\n\n"
        "AVOID:\n"
        "- Excessive empathy phrases like 'that sounds really overwhelming' in every message\n"
        "- Repeating 'I'm not a doctor' more than once per conversation\n"
        "- Asking more than 1-2 clarifying questions before giving useful information\n"
        "- Generic lists of 'possible factors' without committing to the most likely ones\n\n"
    )
    if patient_name:
        system += "The patient's name is " + patient_name + ". Use their name sparingly — maximum once per response, and only when it genuinely helps.\n\n"
    if lang_instruction:
        system += lang_instruction + "\n\n"
    if memory_context:
        system += "WHAT THIS PATIENT HAS TOLD YOU:\n" + memory_context + "\n\n"
    system += (
        "MEDICAL KNOWLEDGE (use this to identify conditions and give specific answers):\n"
        + context
    )
    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}]
    r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.5, max_tokens=1024)
    return r.choices[0].message.content, memory, sources

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
    st.session_state.last_sources = []

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

st.markdown('<div class="header-card"><div style="font-size:2.5rem;">🏥</div><div><div class="header-title">MediChat</div><div class="header-subtitle">Your AI health assistant - chat freely or do a guided symptom check</div></div></div>', unsafe_allow_html=True)

L = LANGUAGES[st.session_state.selected_language]

cm1, cm2 = st.columns(2)
with cm1:
    if st.button(L["free_chat"] + (" (Active)" if st.session_state.mode == "chat" else ""), use_container_width=True):
        st.session_state.mode = "chat"
        st.rerun()
with cm2:
    if st.button(L["symptom_check"] + (" (Active)" if st.session_state.mode == "assessment" else ""), use_container_width=True):
        st.session_state.mode = "assessment"
        st.rerun()

st.markdown('<div class="stats-row"><span class="stat-pill green">RAG - PubMed + MedDialog</span><span class="stat-pill purple">Vision AI Active</span><span class="stat-pill blue">1000 Medical Docs</span><span class="stat-pill orange">Memory Active</span><span class="stat-pill" style="color:#dc2626;border-color:#fecaca;background:#fef2f2;">Emergency Detection</span></div>', unsafe_allow_html=True)
st.markdown('<div class="disclaimer">MediChat provides general health information only - not a substitute for professional medical advice. Always consult a qualified doctor for personal health concerns.</div>', unsafe_allow_html=True)

# Emergency banner - shown whenever emergency keywords detected in this session
if st.session_state.emergency_detected:
    st.markdown(
        '<div class="emergency-banner">'
        '<div class="emergency-title">🚨 This May Be a Medical Emergency</div>'
        '<div class="emergency-text">Based on what you described, you may need immediate medical attention. Please stop and call emergency services now. Do not wait.</div>'
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
                # Show source tags
                msg_sources = msg.get("sources", [])
                if msg_sources:
                    source_tags = "".join(['<span class="source-tag">📚 ' + s + '</span>' for s in msg_sources])
                    st.markdown('<div class="source-row">Grounded in: ' + source_tags + '</div>', unsafe_allow_html=True)

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
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="uploader_" + str(st.session_state.uploader_key))
    if uploaded_image:
        ia, ib, ic = st.columns([1, 2, 1])
        with ib:
            st.image(uploaded_image, caption="Ready for analysis", use_column_width=True)
    st.markdown('<div class="section-label" style="margin-top:0.7rem;">' + L["question_label"] + '</div>', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("", placeholder=L["placeholder"], label_visibility="collapsed")
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc2:
            submit = st.form_submit_button(L["send_btn"])
        with fc3:
            clear = st.form_submit_button(L["clear_btn"])
    st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state.messages = []
        st.session_state.qcount = 0
        st.session_state.feedback = {}
        st.session_state.patient_memory = {"symptoms": [], "conditions": [], "medications": []}
        st.session_state.uploader_key += 1
        st.session_state.emergency_detected = False
        st.session_state.last_sources = []
        st.session_state.patient_name = ""
        st.rerun()

    if submit and (user_input.strip() or uploaded_image):
        st.session_state.qcount += 1
        lang_instruction = LANGUAGES[st.session_state.selected_language]["lang_instruction"]

        # Emergency detection on user input
        if user_input.strip() and detect_emergency(user_input):
            st.session_state.emergency_detected = True

        if uploaded_image:
            st.session_state.messages.append({"role": "user", "type": "image", "content": user_input.strip()})
            with st.spinner("Analysing your image..."):
                uploaded_image.seek(0)
                reply = medichat_vision(user_input, encode_image(uploaded_image), st.session_state.messages, lang_instruction)
            st.session_state.last_sources = ["Vision AI (Llama-4-Scout)"]
        else:
            st.session_state.messages.append({"role": "user", "type": "text", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                name_for_rag = "" if st.session_state.patient_name == "Guest" else st.session_state.patient_name
                reply, memory, sources = medichat_rag(user_input, st.session_state.messages, lang_instruction, name_for_rag)
                st.session_state.patient_memory = memory
                st.session_state.last_sources = sources
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply, "sources": st.session_state.last_sources})
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
                            if current["key"] == "main_symptom" and detect_emergency(opt):
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
                if current["key"] == "main_symptom" and detect_emergency(typed):
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
