# 🏥 MediChat AI

> An AI health companion that lets patients chat in plain language, upload medical PDFs, images and prescriptions, log vitals, sync their calendar, and get personalised guidance that adapts as they use it.

A solo project, designed and built end to end by **Ahkeeshan Sarvananthan**.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?logo=streamlit&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-FAISS%20%2B%20MiniLM-4B8BBE)
![Vision](https://img.shields.io/badge/Multimodal-Vision%20%2B%20OCR-6E40C9)

---

## What MediChat does

- **Natural-language chat** about symptoms, conditions, medications and general health questions, with retrieval over real medical literature.
- **Multimodal uploads** for medical PDFs (lab reports, specialist letters), images (scans, photos) and handwritten prescriptions. The uploaded image is shown inline in the conversation and embedded into every exported PDF.
- **Vision AI** for chest X-rays, skin lesions and scans. It gives a localised reading (which lung, which zone), a structured differential that includes serious but treatable causes, and concrete next steps. You tap Analyze on the Home page, the chat opens instantly with the image and a thinking spinner, and the structured analysis streams in.
- **Prescription Reader** with OCR-grade transcription of handwritten scripts. It detects multiple medications per script (for example Augmentin, Enzoflam, Pan-D and advice items), expands medical Latin (BD, TDS, mane, nocte, Mitte), maps Australian PBS brands to generics, grades per-field confidence, cross-checks against a known-drug list, and falls back through a provider chain when the lead model refuses or hallucinates.
- **Health Records** where every uploaded report becomes a searchable record with an AI-generated summary, plus View document and View AI summary options. Records persist across chats and feed back into replies.
- **Symptoms Checker**, a guided multi-step assessment that produces a shareable Doctor Visit Summary PDF.
- **Health Overview** with daily vitals (sleep, water, steps, heart rate, mood), 7-day sparkline trends, and tighter clinical thresholds (100 BPM reads "High" not "Normal", 90 to 99 reads "Elevated"). A today-vs-yesterday comparator shows red warning chips when heart rate jumps by 20 bpm or more, sleep drops by 2 hours or more, activity collapses, or hydration crashes.
- **AI Insights**: auto-detected health trends, care gaps, family-risk reminders, and personalised observations grounded in your logged data.
- **Adaptive Memory** that auto-extracts allergies, medications, conditions, symptoms and appointments from chat and quietly adds them to your profile. Past chats are embedded so MediChat can recall "you asked about this last month", and its response style adapts to your message length and thumbs feedback.
- **Calendar Sync**: paste your Google, Apple or Outlook calendar ICS URL once and MediChat keeps your upcoming health appointments in sync automatically.
- **Second Opinion**, an optional pass where a different frontier model reviews the first model's answer before it ships.
- **PDF exports** (Conversation Export, Doctor Visit Summary, Patient Medical Record), each with a polished branded header, tinted chat bubbles, embedded images and page numbers. Timestamps use the viewer's local time.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend / runtime | Streamlit 1.50 |
| AI backends (text, in order) | OpenAI GPT-4o-mini, then Anthropic Claude Haiku 4.5, then Groq Llama-3.3 |
| Vision (X-ray and image analysis) | OpenAI GPT-4o (`detail: "high"`), then Claude vision, then Groq Llama-4 Scout |
| Prescription OCR | OpenAI GPT-4o, then Claude vision, then Groq Llama-4 Scout (with refusal and hallucination guards) |
| Retrieval (RAG) | sentence-transformers (`all-MiniLM-L6-v2`) and FAISS |
| Knowledge sources | 25,000 documents: about 12,500 PubMed research abstracts (`qiaojin/PubMedQA`, `pqa_artificial`) and about 12,500 MedDialog doctor-patient dialogues |
| Persistence | Firebase Firestore (per-user profile, chats, records, settings) |
| Caching | Short-TTL per-user memoization on hot Firestore reads |
| Calendar import | iCalendar (ICS) parser for Google, Apple and Outlook |
| Voice input | `streamlit-mic-recorder` and Groq Whisper, auto-transcribe on stop |
| PDF generation | fpdf2 with embedded brand logo and uploaded images |
| PDF reading | pypdf |
| Image preprocessing | Pillow (autocontrast, sharpness, 2048px max for handwriting) |
| Language | Python 3.11+ |
| Deployment | Streamlit Cloud |

## Run locally

```bash
pip install -r requirements.txt

# Add your secrets at .streamlit/secrets.toml (see secrets.toml.example)
streamlit run app.py
```

`.streamlit/secrets.toml` is gitignored, so your API keys are never committed.

Required secret keys:

```toml
OPENAI_API_KEY     = "sk-..."
ANTHROPIC_API_KEY  = "sk-ant-..."
GROQ_API_KEY       = "gsk_..."
PROFILE_SALT       = "<random-32-byte-hex>"

[firebase_service_account]
# Paste your Firebase Admin SDK service account JSON here
```

## Note

MediChat is a personal project I built during my Master of Information Technology. It is not a registered medical device and is not a substitute for professional medical advice, diagnosis or treatment. Always consult a qualified healthcare professional. In an emergency call 000 (AU), 911 (US) or 112 (EU).

## Author

Designed and built solo by **Ahkeeshan Sarvananthan**.

- GitHub: [@akeeshh](https://github.com/akeeshh)
- LinkedIn: [linkedin.com/in/akeeshh](https://linkedin.com/in/akeeshh)

© 2026 Ahkeeshan Sarvananthan. All rights reserved.
