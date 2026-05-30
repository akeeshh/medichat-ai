# 🏥 MediChat AI — Multimodal Clinical Companion

> A Streamlit-based AI health companion that lets patients chat in plain
> language, upload medical PDFs / images / prescriptions, log vitals,
> sync their calendar, and receive personalised guidance that adapts as
> they use it.

## 👥 Team — Group 3 | ICT654 Final Year Project | SISTC

| Name | Student ID | Role |
|------|-----------|------|
| Ahkeeshan Sarvananthan | S20250214 | Group Leader |
| Thanusha Shakthivelu | S20242820 | Member |
| Gurpreet Singh | S20242457 | Member |
| Yehan Ayesh Kumarathilake | S20242471 | Member |
| Muskan Garg | S20250201 | Member |

**Supervisor:** Dr. Amoakoh Gyasi-Agyei

---

## 🧠 What MediChat Does

- **Natural-language chat** about symptoms, conditions, medications, and
  general health questions, with retrieval over real medical literature.
- **Multimodal uploads** — medical PDFs (lab reports, specialist
  letters), images (scans, photos), and handwritten prescriptions.
- **Prescription Reader** — OCR-style transcription of handwritten
  scripts with structured output (medication, dose, frequency, route),
  confidence grading, and an Australian drug-name cross-check.
- **Health Records** — every upload becomes a searchable record with
  AI-generated summary the chat can reference later.
- **Symptoms Checker** — guided multi-step assessment producing a
  shareable Doctor Visit Summary PDF.
- **Health Overview** — daily vitals (sleep, water, steps, heart rate,
  mood) with 7-day sparkline trends.
- **AI Insights** — auto-detected health trends, care gaps, family-risk
  reminders, and personalised observations grounded in your logged data.
- **Adaptive Memory** — MediChat auto-extracts allergies, medications,
  conditions, symptoms, and appointments from chat and silently adds
  them to your profile. Past chats are embedded so MediChat can recall
  "you asked about this last month". Response style learns from your
  message length + thumbs feedback.
- **Calendar Sync** — paste your Google / Apple / Outlook calendar's
  ICS URL once; MediChat keeps a list of upcoming health-related
  appointments in sync automatically.
- **MediChat Verify** — optional second-opinion pass where a different
  model reviews the first model's answer before showing it.
- **Doctor Visit Summary PDF** — one-page handover summary for any GP.

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / runtime | Streamlit 1.50 |
| AI backends (with fallback) | OpenAI GPT-4o-mini → Anthropic Claude Haiku 4.5 → Groq Llama-3.3 |
| Vision (prescription / image analysis) | Claude vision + Groq Llama Scout |
| Retrieval (RAG) | sentence-transformers (`all-MiniLM-L6-v2`) + FAISS |
| Knowledge sources | PubMedQA + MedDialog (HuggingFace `datasets`) |
| Persistence | Firebase Firestore (per-user profile, chats, records, settings) |
| Calendar import | iCalendar (ICS) parser — Google / Apple / Outlook |
| Voice input | `streamlit-mic-recorder` + Groq Whisper |
| PDF generation | fpdf2 |
| PDF reading | pypdf |
| Image preprocessing | Pillow |
| Language | Python 3.11 |
| Version Control | GitHub |

## 🚀 Run Locally

```bash
pip install -r requirements.txt

# Add your secrets at .streamlit/secrets.toml — see secrets.toml.example
streamlit run app.py
```

`.streamlit/secrets.toml` is gitignored — never commit your API keys.

## 📅 Project Timeline

| Weeks | Phase |
|-------|-------|
| 1–2 | Planning & Setup |
| 3–5 | Model Integration & RAG Pipeline |
| 6–8 | Web UI Development |
| 9–10 | Testing & Evaluation |
| 11–12 | Final Report & Demo |

## ⚠️ Important — Research Preview

MediChat is a final-year capstone project. It is **not** a registered
medical device and is **not** a substitute for clinical advice,
diagnosis, or treatment. Always consult a qualified healthcare
professional for medical concerns. In an emergency call **000** (AU) /
**911** (US) / **112** (EU).

------

*ICT654 — SISTC, Melbourne, Australia*
