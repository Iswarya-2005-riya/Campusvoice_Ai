"""
CampusVoice AI - AI Engine
--------------------------
Lightweight, dependency-free NLP engine that handles:
  1. Category classification   (which department should this go to?)
  2. Urgency / priority scoring (how fast does this need attention?)
  3. Sentiment detection        (how upset is the student?)
  4. Duplicate/similar complaint detection (cluster repeated issues)

This uses keyword-weighted scoring + text similarity instead of a heavy
ML model, so it runs instantly with zero external dependencies and no
internet access. It's designed to be swapped out for a real transformer
model (HuggingFace) or an LLM API call later -- see `classify_with_llm()`
at the bottom for where that would plug in.
"""

import re
import difflib
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. CATEGORY KEYWORDS
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Hostel": [
        "hostel", "room", "roommate", "warden", "bed", "mattress", "washroom",
        "bathroom", "water supply", "leakage", "leak", "electricity", "power cut",
        "fan", "light not working", "door lock", "key", "cleanliness", "pest",
        "cockroach", "bedbug"
    ],
    "Mess/Food": [
        "mess", "food", "canteen", "cafeteria", "meal", "breakfast", "lunch",
        "dinner", "hygiene", "insect in food", "stale", "rotten", "quality of food",
        "menu", "overcharging", "mess bill"
    ],
    "Academics": [
        "professor", "faculty", "lecture", "class", "exam", "grading", "marks",
        "syllabus", "assignment", "attendance", "timetable", "course", "grade",
        "result", "re-evaluation", "internal marks"
    ],
    "Harassment/Ragging": [
        "ragging", "ragged", "rag by", "harass", "bully", "bullying", "abuse",
        "threat", "assault", "discriminat", "molest", "stalk", "unsafe",
        "misbehav", "inappropriate touch", "senior forced", "feel unsafe"
    ],
    "IT/Infrastructure": [
        "wifi", "wi-fi", "internet", "network", "server", "website", "portal",
        "login", "password reset", "lms", "projector", "computer lab", "software",
        "printer", "id card"
    ],
    "Library": [
        "library", "book", "librarian", "fine", "return date", "reading room",
        "journal", "subscription"
    ],
    "Transport": [
        "bus", "transport", "shuttle", "driver", "route", "bus pass", "parking"
    ],
    "Finance/Fees": [
        "fee", "fees", "scholarship", "refund", "payment", "invoice", "receipt",
        "bank", "transaction failed"
    ],
}

# ---------------------------------------------------------------------------
# 2. URGENCY / SAFETY KEYWORDS (weighted)
# ---------------------------------------------------------------------------
URGENCY_KEYWORDS = {
    # word/phrase : weight (higher = more urgent)
    "ragging": 10, "harass": 10, "assault": 10, "molest": 10, "threat": 9,
    "unsafe": 8, "fire": 10, "injury": 9, "injured": 9, "accident": 9,
    "electric shock": 9, "short circuit": 8, "gas leak": 10, "suicide": 10,
    "emergency": 9, "urgent": 6, "immediately": 5, "not working": 2,
    "since a week": 4, "since last month": 5, "repeatedly": 4, "again and again": 4,
}

NEGATIVE_SENTIMENT_WORDS = [
    "angry", "furious", "disgusted", "worst", "terrible", "horrible", "useless",
    "pathetic", "disappointed", "frustrat", "unacceptable", "fed up", "sick of",
    "never", "no action", "ignored", "waste of"
]

CRITICAL_CATEGORIES = {"Harassment/Ragging"}


def _score_keywords(text, keyword_dict):
    """Return dict of {key: count_found} for a keyword->weight dict, matched on substrings."""
    text_l = text.lower()
    hits = {}
    for kw, weight in keyword_dict.items():
        if kw in text_l:
            hits[kw] = weight
    return hits


def classify_category(text):
    """Return the best-matching category based on keyword overlap."""
    text_l = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_l)
        if score:
            scores[category] = score

    if not scores:
        return "General/Uncategorized", 0.0

    # Safety-critical categories win on any hit, even if a more "generic" word
    # (e.g. "room", "hostel") scored higher elsewhere — we never want a ragging
    # complaint mis-routed to Hostel just because it also mentions a room.
    for critical_cat in CRITICAL_CATEGORIES:
        if critical_cat in scores:
            total = sum(scores.values())
            confidence = round(scores[critical_cat] / total, 2) if total else 0.0
            return critical_cat, confidence

    best_category = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = round(scores[best_category] / total, 2) if total else 0.0
    return best_category, confidence


def detect_sentiment(text):
    """Very simple lexicon-based sentiment: returns 'Negative', 'Neutral' or label + score."""
    text_l = text.lower()
    neg_hits = sum(1 for w in NEGATIVE_SENTIMENT_WORDS if w in text_l)
    exclamations = text.count("!")
    caps_words = sum(1 for w in text.split() if len(w) > 3 and w.isupper())

    negativity_score = neg_hits * 2 + exclamations + caps_words
    if negativity_score >= 4:
        return "Highly Negative", negativity_score
    elif negativity_score >= 1:
        return "Negative", negativity_score
    return "Neutral", negativity_score


def compute_urgency(text, category):
    """
    Combine urgency keywords + category + sentiment into a 0-100 urgency score
    and a human label: Low / Medium / High / Critical.
    """
    hits = _score_keywords(text, URGENCY_KEYWORDS)
    keyword_score = sum(hits.values())

    sentiment_label, sentiment_score = detect_sentiment(text)
    sentiment_boost = min(sentiment_score * 2, 10)

    category_boost = 25 if category in CRITICAL_CATEGORIES else 0

    raw_score = keyword_score * 4 + sentiment_boost + category_boost
    urgency_score = min(int(raw_score), 100)

    if urgency_score >= 70 or category in CRITICAL_CATEGORIES:
        label = "Critical"
    elif urgency_score >= 40:
        label = "High"
    elif urgency_score >= 15:
        label = "Medium"
    else:
        label = "Low"

    return {
        "score": urgency_score,
        "label": label,
        "matched_keywords": list(hits.keys()),
        "sentiment": sentiment_label,
    }


def is_duplicate(new_text, existing_complaints, threshold=0.6):
    """
    Compare new_text against a list of existing complaint texts using
    difflib's SequenceMatcher (ratio of similarity, 0-1).
    Returns (is_dup: bool, matched_id: int|None, similarity: float)
    """
    best_ratio = 0.0
    best_id = None
    new_text_l = new_text.lower().strip()

    for complaint in existing_complaints:
        existing_text_l = complaint["description"].lower().strip()
        ratio = difflib.SequenceMatcher(None, new_text_l, existing_text_l).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = complaint["id"]

    return (best_ratio >= threshold, best_id, round(best_ratio, 2))


def suggest_self_help(category, text):
    """Return a quick self-help tip for common low-urgency issues, or None."""
    text_l = text.lower()
    tips = {
        "wifi": "Try: 1) Toggle Wi-Fi off/on 2) Forget & rejoin the network "
                "'CampusNet' 3) Restart your router if using a personal one. "
                "If it still fails, this complaint will be routed to IT.",
        "password reset": "You can self-reset your portal password via "
                           "Settings > Forgot Password before this ticket is picked up.",
        "printer": "Check the printer queue isn't jammed with a stuck print job. "
                   "If it's a hardware fault, this will go to IT/Infrastructure.",
        "library fine": "Fines can be checked and paid directly from your Library "
                         "portal account under 'My Fines'.",
    }
    for key, tip in tips.items():
        if key in text_l:
            return tip
    return None


def analyze_complaint(text, existing_complaints=None):
    """
    Main entry point: run the full pipeline on a new complaint description.
    existing_complaints: list of dicts like {"id": int, "description": str}
    """
    existing_complaints = existing_complaints or []

    category, confidence = classify_category(text)
    urgency = compute_urgency(text, category)
    dup_flag, dup_id, similarity = is_duplicate(text, existing_complaints)
    self_help = suggest_self_help(category, text)

    return {
        "category": category,
        "category_confidence": confidence,
        "urgency_score": urgency["score"],
        "urgency_label": urgency["label"],
        "sentiment": urgency["sentiment"],
        "matched_urgency_keywords": urgency["matched_keywords"],
        "is_duplicate": dup_flag,
        "duplicate_of_id": dup_id if dup_flag else None,
        "similarity": similarity,
        "self_help_tip": self_help,
        "analyzed_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# OPTIONAL: swap-in point for a real LLM instead of the rule-based engine.
# Uncomment and wire up an API key to use this instead of analyze_complaint().
# ---------------------------------------------------------------------------
def classify_with_llm(text):
    """
    Placeholder showing where you'd plug in a real LLM call (e.g. Anthropic /
    OpenAI API) for higher-accuracy classification instead of keyword matching.
    Left unimplemented so the project runs with zero API keys / internet.
    """
    raise NotImplementedError(
        "Wire this up to your preferred LLM API (Anthropic/OpenAI) if you want "
        "model-based classification instead of the built-in rule-based engine."
    )
