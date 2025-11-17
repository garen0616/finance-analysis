import os
import logging
from typing import Dict, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

logger = logging.getLogger(__name__)

def _extractive_summary(text: str, sentences=6):
    if not text:
        return {"title": "No transcript", "bullets": []}
    parts = [p.strip() for p in text.split(".") if p.strip()]
    # drop very short utterances
    parts = [p for p in parts if len(p.split()) >= 5]
    if not parts:
        return {"title": "No transcript", "bullets": []}
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(parts)
    scores = (X.sum(axis=1) / (X != 0).sum(axis=1)).A.ravel()
    ranked = [p for _, p in sorted(zip(scores, parts), key=lambda t: t[0], reverse=True)]
    n = min(sentences, len(ranked))
    chosen = ranked[:n]
    keywords = [w for w, _ in Counter(" ".join(parts).lower().split()).most_common(8)]
    return {"title": "Executive Summary", "bullets": chosen, "keywords": keywords}

def _openai_summary_http(text: str, model: str | None):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OpenAI key")
    url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    payload = {
        "model": model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Summarize as a concise sell-side earnings recap with 6-10 bullets and one-sentence overview."},
            {"role": "user", "content": text[:12000]},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    bullets = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
    return {"title": "Executive Summary", "bullets": bullets}

def summarize_text(text: str, rag: Dict[str, Any] | None, profile: Dict[str, Any]):
    openai_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_SUMMARY_MODEL") or "gpt-5"
    if openai_key:
        try:
            return _openai_summary_http(text, model)
        except Exception as exc:
            logger.warning("OpenAI summary failed (fallback to extractive): %s", exc)
    base = _extractive_summary(text)
    if rag and rag.get("insights"):
        base["bullets"] = base.get("bullets", []) + [i.get("text", "") for i in rag["insights"][:3]]
    return base
