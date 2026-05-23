import os
import re
import difflib
from database import _safe_secret

AU_TOP_DRUGS_FALLBACK = {
    "amoxicillin", "amoxicillin clavulanate", "azithromycin", "cephalexin", "doxycycline", "trimethoprim",
    "metformin", "gliclazide", "empagliflozin", "sitagliptin", "insulin glargine",
    "atorvastatin", "rosuvastatin", "simvastatin", "ezetimibe",
    "amlodipine", "perindopril", "lisinopril", "ramipril", "losartan", "valsartan", "metoprolol",
    "aspirin", "clopidogrel", "apixaban", "rivaroxaban", "warfarin",
    "salbutamol", "budesonide", "fluticasone", "tiotropium", "montelukast",
    "levothyroxine", "prednisone", "prednisolone", "omeprazole", "esomeprazole", "pantoprazole",
    "ibuprofen", "naproxen", "diclofenac", "paracetamol", "codeine", "tramadol",
    "sertraline", "escitalopram", "fluoxetine", "mirtazapine", "venlafaxine",
    "quetiapine", "olanzapine", "risperidone", "lithium",
    "ondansetron", "metoclopramide", "loperamide", "dimenhydrinate",
    "nitrofurantoin", "ciprofloxacin", "valaciclovir", "acyclovir",
}

def load_known_drugs():
    csv_path = _safe_secret("AU_DRUGS_CSV_PATH", os.environ.get("AU_DRUGS_CSV_PATH", ""))
    known = set(AU_TOP_DRUGS_FALLBACK)
    if not csv_path:
        return known
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                token = (line or "").strip().lower()
                if token and token not in {"drug", "name", "medication"}:
                    known.add(token)
    except Exception as e:
        print("Known-drug list load failed:", e)
    return known

KNOWN_DRUGS = load_known_drugs()

def validate_drug_name(reading):
    candidate = (reading or "").lower().strip()
    if not candidate:
        return {"match": "none"}
    if candidate in KNOWN_DRUGS:
        return {"match": "exact", "drug": candidate}
    close = difflib.get_close_matches(candidate, KNOWN_DRUGS, n=3, cutoff=0.76)
    if close:
        return {"match": "close", "candidates": close}
    return {"match": "none"}

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

EMERGENCY_SYMPTOM_CLUSTERS = [
    {
        "name": "Possible cardiac event",
        "required": [["chest", "chest pressure", "chest tight"]],
        "any_of": [["breath", "breathless", "out of breath"], ["arm pain", "jaw pain", "left arm"], ["sweat", "sweating"], ["nausea", "vomit"]],
        "min_any": 2,
    },
    {
        "name": "Possible stroke",
        "required": [],
        "any_of": [["face droop", "drooping"], ["slurred", "speech"], ["weakness one side", "numb one side"], ["sudden confusion"], ["sudden severe headache"]],
        "min_any": 2,
    },
    {
        "name": "Possible severe allergic reaction",
        "required": [],
        "any_of": [["swelling"], ["throat", "tongue"], ["hives", "rash all over"], ["difficulty breathing", "trouble breathing"]],
        "min_any": 3,
    },
    {
        "name": "Possible syncope/pre-syncope episode",
        "required": [],
        "any_of": [["faint", "faintish", "feel faint"], ["dizzy", "lightheaded"], ["blank", "going blank"], ["nausea"], ["heart racing", "heartbeat raises", "palpitations"], ["breath", "breathless"]],
        "min_any": 3,
    },
    {
        "name": "Possible severe asthma exacerbation",
        "required": [["asthma", "asthmatic"]],
        "any_of": [["can't breathe", "cant breathe", "struggling to breathe"], ["inhaler not working", "inhaler isn't helping"], ["blue lips", "blue fingers"], ["cannot speak", "can't talk"]],
        "min_any": 1,
    },
    {
        "name": "Possible hyperglycemic crisis",
        "required": [["diabetes", "diabetic"]],
        "any_of": [["excessive thirst"], ["frequent urination"], ["confusion"], ["fruity breath"], ["vomiting"], ["extreme fatigue"]],
        "min_any": 3,
    },
]

META_INDICATORS = [
    "streamlit", "redeploy", "redeploying", "deployment", "deploy",
    "tier 1", "tier 2", "tier 3", "tier 4", "tier 5",
    "test path", "test plan", "test scenario", "test case",
    "feature", "merging", "merged", "commit", "git push", "branch",
    "github", "claude.ai/code", "session_01", "session_state",
    "pull request", " pr #", "rebase",
    "what you'll see", "what you’ll see",
    "shipped to main", "redeploy in",
]

def is_meta_text(text):
    if not text:
        return False
    t = text.lower()
    score = sum(1 for ind in META_INDICATORS if ind in t)
    has_md = ("**" in t) or ("###" in t) or ("```" in t)
    if has_md:
        score += 1
    if len(t) > 1500:
        score += 1
    return score >= 2

def detect_emergency(text, conversation_text=""):
    if is_meta_text(text):
        return False, None
    if not text and not conversation_text:
        return False, None
    combined = (text + " " + conversation_text).lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in combined:
            return True, "Emergency keyword detected"
    for cluster in EMERGENCY_SYMPTOM_CLUSTERS:
        required_met = True
        for req_group in cluster["required"]:
            if not any(term in combined for term in req_group):
                required_met = False
                break
        if not required_met:
            continue
        matches = 0
        for any_group in cluster["any_of"]:
            if any(term in combined for term in any_group):
                matches += 1
        if matches >= cluster["min_any"]:
            return True, cluster["name"]
    return False, None

URGENT_KEYWORDS = [
    "severe pain", "worst pain", "intense pain", "unbearable pain",
    "high fever", "fever over 39", "fever 40", "very high temperature",
    "vomiting blood", "blood in stool", "blood in urine", "coughing up blood",
    "cannot keep anything down", "can't keep food down",
    "severe dehydration", "haven't urinated in", "no urine for",
    "severe headache that won't go", "thunderclap headache",
    "vision changes", "double vision", "lost vision",
    "new confusion", "disoriented", "very confused",
    "severe abdominal pain", "rigid abdomen",
    "rapid heartbeat", "racing heart for hours",
    "severe dizziness", "cannot stand",
    "broken bone", "deformed limb", "bone visible",
    "deep cut", "deep wound", "wound won't stop bleeding",
    "burn larger than", "blistering burn",
    "severe asthma attack", "wheezing badly",
    "severe panic", "cannot stop shaking",
    "pregnancy bleeding", "severe pregnancy pain",
]

CONCERN_KEYWORDS = [
    "fever for", "persistent fever", "fever won't go", "fever lasting",
    "headache for days", "persistent headache", "daily headaches",
    "cough for weeks", "persistent cough", "cough lasting",
    "rash spreading", "rash worse", "growing rash",
    "weight loss", "losing weight", "lost weight",
    "tired all the time", "exhausted constantly", "no energy for weeks",
    "trouble sleeping for", "insomnia for",
    "persistent pain", "pain for days", "pain getting worse",
    "frequent urination", "burning urination", "painful urination",
    "diarrhoea for", "diarrhea for", "constipation for",
    "swelling that won't", "lump that's growing", "new lump",
    "bleeding gums often", "easy bruising",
    "new mole", "mole changing", "changing mole",
    "irregular periods", "missed period",
    "anxious all the time", "feeling depressed", "low mood for weeks",
    "joint pain for", "stiff joints",
    "shortness of breath on stairs", "out of breath easily",
]

def assess_triage_tier(text, conversation_text="", memory=None):
    text_l = (text or "").lower()
    conv_l = (conversation_text or "").lower()
    combined = (text_l + " " + conv_l).strip()
    memory = memory or {}

    if is_meta_text(text):
        return {
            "tier": 5, "label": "Self-care appropriate",
            "icon": "⚪", "color": "#64748b",
            "bg": "linear-gradient(135deg,#64748b,#475569)",
            "next_step": "",
            "reasons": [],
        }

    is_emerg, emerg_reason = detect_emergency(text, conversation_text)
    if is_emerg:
        return {
            "tier": 1,
            "label": "Emergency",
            "icon": "🔴",
            "color": "#dc2626",
            "bg": "linear-gradient(135deg,#dc2626,#991b1b)",
            "next_step": "Call 000 (Australia) or your local emergency number now. Do not wait. Do not drive yourself.",
            "reasons": [emerg_reason or "Emergency pattern detected"],
        }

    matched_urgent = [kw for kw in URGENT_KEYWORDS if kw in combined]
    if matched_urgent:
        return {
            "tier": 2,
            "label": "Urgent care today",
            "icon": "🟠",
            "color": "#ea580c",
            "bg": "linear-gradient(135deg,#ea580c,#c2410c)",
            "next_step": "See a GP, urgent care clinic, or ED today, ideally within 2-4 hours.",
            "reasons": matched_urgent[:3],
        }

    matched_concern = [kw for kw in CONCERN_KEYWORDS if kw in combined]
    has_chronic = bool(memory.get("conditions"))
    if matched_concern or (has_chronic and any(s in combined for s in ["worse", "new symptom", "different"])):
        reasons = matched_concern[:3] if matched_concern else ["Chronic condition + new or worsening symptom"]
        return {
            "tier": 3,
            "label": "GP within 24-48 hrs",
            "icon": "🟡",
            "color": "#ca8a04",
            "bg": "linear-gradient(135deg,#ca8a04,#a16207)",
            "next_step": "Book a GP appointment for today or tomorrow. Watch for worsening signs.",
            "reasons": reasons,
        }

    has_symptoms = bool(memory.get("symptoms")) or any(w in combined for w in ["pain", "ache", "sore", "tired", "dizzy", "nausea"])
    if has_symptoms:
        return {
            "tier": 4,
            "label": "GP within a week",
            "icon": "🟢",
            "color": "#16a34a",
            "bg": "linear-gradient(135deg,#16a34a,#15803d)",
            "next_step": "Book a routine GP appointment in the next few days to discuss your symptoms.",
            "reasons": (memory.get("symptoms") or [])[:3] or ["Active symptoms reported"],
        }

    return {
        "tier": 5,
        "label": "Self-care appropriate",
        "icon": "⚪",
        "color": "#64748b",
        "bg": "linear-gradient(135deg,#64748b,#475569)",
        "next_step": "Self-care and monitoring is reasonable. Reach out if anything new or worsening.",
        "reasons": [],
    }

DRUG_INTERACTIONS = {
    "antihistamine": {
        "drugs": ["dramamine", "dimenhydrinate", "bonine", "meclizine", "benadryl", "diphenhydramine", "chlorpheniramine", "promethazine"],
        "conditions": ["asthma", "copd", "glaucoma", "bph", "enlarged prostate"],
        "warning": "Antihistamines can dry respiratory secretions (worsening asthma/COPD), raise intraocular pressure (glaucoma), or cause urinary retention (BPH)."
    },
    "nsaid": {
        "drugs": ["ibuprofen", "advil", "nurofen", "naproxen", "aleve", "aspirin", "diclofenac", "voltaren"],
        "conditions": ["hypertension", "high blood pressure", "kidney", "renal", "ulcer", "gastritis", "asthma", "heart failure", "anticoagulant", "warfarin", "blood thinner"],
        "warning": "NSAIDs can raise blood pressure, worsen kidney function, cause GI bleeding (especially with ulcers or blood thinners), and trigger bronchospasm in some asthmatics."
    },
    "decongestant": {
        "drugs": ["pseudoephedrine", "sudafed", "phenylephrine", "oxymetazoline"],
        "conditions": ["hypertension", "high blood pressure", "heart disease", "arrhythmia", "thyroid", "hyperthyroid", "diabetes", "glaucoma", "bph"],
        "warning": "Decongestants raise blood pressure and heart rate, destabilise cardiac rhythm, worsen hyperthyroidism, and can affect glaucoma or urinary retention."
    },
    "paracetamol_high_dose": {
        "drugs": ["paracetamol", "acetaminophen", "panadol", "tylenol"],
        "conditions": ["liver", "hepatitis", "cirrhosis", "heavy alcohol", "alcoholic"],
        "warning": "High-dose or chronic paracetamol use can cause severe liver damage, especially with pre-existing liver disease or heavy alcohol use."
    },
    "bismuth": {
        "drugs": ["pepto-bismol", "bismuth subsalicylate"],
        "conditions": ["aspirin allergy", "kidney", "renal", "bleeding disorder", "anticoagulant", "warfarin"],
        "warning": "Bismuth subsalicylate contains salicylate (aspirin-like), unsafe with aspirin allergy, kidney disease, or blood thinners."
    },
    "ppi_h2": {
        "drugs": ["omeprazole", "esomeprazole", "pantoprazole", "ranitidine", "famotidine", "zantac"],
        "conditions": ["osteoporosis", "kidney", "c. diff", "magnesium deficiency"],
        "warning": "Long-term PPI use can reduce calcium/magnesium absorption (bone risk), affect kidney function, and increase C. diff infection risk."
    },
}

def check_drug_interactions(response_text, memory):
    if not response_text:
        return []
    text_lower = response_text.lower()
    conditions_text = " ".join(memory.get("conditions", [])).lower() + " " + " ".join(memory.get("medications", [])).lower()
    if not conditions_text.strip():
        return []
    
    # Tokenize conditions/medications to match fuzzily
    words_in_memory = re.findall(r"\b[a-z]{3,}\b", conditions_text)
    words_in_response = re.findall(r"\b[a-z]{3,}\b", text_lower)
    
    alerts = []
    for class_name, info in DRUG_INTERACTIONS.items():
        drug_hit = None
        for d in info["drugs"]:
            if d in text_lower:
                drug_hit = d
                break
            # Fuzzy match word-by-word
            close_matches = difflib.get_close_matches(d, words_in_response, n=1, cutoff=0.85)
            if close_matches:
                drug_hit = d
                break
                
        if not drug_hit:
            continue
            
        condition_hits = []
        for c in info["conditions"]:
            if c in conditions_text:
                condition_hits.append(c)
            else:
                close_cond = difflib.get_close_matches(c, words_in_memory, n=1, cutoff=0.85)
                if close_cond:
                    condition_hits.append(c)
                    
        if condition_hits:
            alerts.append({
                "drug": drug_hit.title(),
                "conditions": list(set(condition_hits)),
                "warning": info["warning"]
            })
    return alerts
