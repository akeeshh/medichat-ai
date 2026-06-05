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
  letters), images (scans, photos), and handwritten prescriptions. The
  uploaded image is shown inline in the conversation and embedded into
  every exported PDF.
- **Vision AI** — chest X-rays, skin lesions, scans. Localised reading
  (which lung / which zone), structured differential including
  serious-but-treatable causes, and concrete next-step guidance. Patient
  taps Analyze on Home → chat page opens instantly with the image and a
  thinking spinner, then the structured analysis streams in.
- **Prescription Reader** — OCR-grade transcription of handwritten
  scripts. Detects multiple medications per script (Augmentin + Enzoflam
  + Pan-D + advice items, etc.), expands medical Latin (BD / TDS / mane
  / nocte / Mitte), maps Australian PBS brands to generics, grades
  per-field confidence, cross-checks against a known-drug list, and
  falls back through a provider chain when the lead model refuses or
  hallucinates.
- **Health Records** — every uploaded report becomes a searchable record
  with AI-generated summary and View document / View AI summary
  affordances. Records persist across chats and feed back into chat
  replies.
- **Symptoms Checker** — guided multi-step assessment producing a
  shareable Doctor Visit Summary PDF.
- **Health Overview** — daily vitals (sleep, water, steps, heart rate,
  mood) with 7-day sparkline trends. Tighter clinical thresholds:
  100 BPM reads "High" not "Normal", 90-99 reads "Elevated". A
  today-vs-yesterday comparator surfaces red warning chips when heart
  rate jumps ≥ 20 bpm, sleep drops ≥ 2 h, activity collapses, or
  hydration crashes.
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
- **MediChat Second Opinion** — optional second-opinion pass where a
  different frontier model reviews the first model's answer before it
  ships, stacked title over subtitle on the result card.
- **PDF exports** — Conversation Export, Doctor Visit Summary, and
  Patient Medical Record. All three share a polished header with the
  MediChat brand bar, tagline, pill-chip section labels, tinted chat
  bubbles (blue for user, teal for MediChat), embedded uploaded images,
  and page numbers in the footer. Timestamps reflect the viewer's local
  time, not the server's UTC.

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / runtime | Streamlit 1.50 |
| AI backends (text, in order) | OpenAI GPT-4o-mini → Anthropic Claude Haiku 4.5 → Groq Llama-3.3 |
| Vision (X-ray / image analysis) | OpenAI GPT-4o (`detail: "high"`) → Claude vision → Groq Llama-4 Scout |
| Prescription OCR | OpenAI GPT-4o → Claude vision → Groq Llama-4 Scout (with refusal + hallucination guards) |
| Retrieval (RAG) | sentence-transformers (`all-MiniLM-L6-v2`) + FAISS |
| Knowledge sources | 25,000 documents — ~12,500 PubMed research abstracts (`qiaojin/PubMedQA` config `pqa_artificial`) + ~12,500 MedDialog doctor-patient dialogues (HuggingFace `datasets`) |
| Persistence | Firebase Firestore (per-user profile, chats, records, settings) |
| Per-process caching | Short-TTL memoization (per-user-keyed) on hot Firestore reads |
| Calendar import | iCalendar (ICS) parser — Google / Apple / Outlook |
| Voice input | `streamlit-mic-recorder` + Groq Whisper with auto-transcribe-on-stop |
| PDF generation | fpdf2 with embedded brand logo + uploaded images |
| PDF reading | pypdf |
| Image preprocessing | Pillow (autocontrast + sharpness + 2048px max for handwriting) |
| Language | Python 3.11+ |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

## ⚡ Recent Polish (June 2026)

- Instant nav-click transitions: clicking any sidebar item hides the old
  page and shows a brand-blue spinner the moment the click registers;
  the new page fades in over 90 ms when its DOM is ready.
- Mobile (≤ 768 px) takes the sidebar out of the flex flow so the main
  content fills the entire viewport; no toggle can summon the drawer.
- Care Circle (bilateral partner linking) backend retained but the
  in-app entry surface was removed for the keynote demo.

## 🚀 Run Locally

```bash
pip install -r requirements.txt

# Add your secrets at .streamlit/secrets.toml — see secrets.toml.example
streamlit run app.py
```

`.streamlit/secrets.toml` is gitignored — never commit your API keys.

Required secret keys:

```toml
OPENAI_API_KEY     = "sk-..."
ANTHROPIC_API_KEY  = "sk-ant-..."
GROQ_API_KEY       = "gsk_..."
PROFILE_SALT       = "<random-32-byte-hex>"

[firebase_service_account]
# Paste your Firebase Admin SDK service account JSON here
```

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
