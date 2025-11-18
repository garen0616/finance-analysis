import os
import logging
from typing import Dict, Any
from collections import Counter, defaultdict
import requests

logger = logging.getLogger(__name__)

def _extractive_summary(text: str, sentences: int = 6):
    if not text:
        return {"title": "No transcript", "bullets": []}
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 5]
    if not parts:
        return {"title": "No transcript", "bullets": []}
    # simple term frequency scoring (no sklearn)
    words = " ".join(parts).lower().split()
    stop = set(["the","and","an","a","of","to","in","for","on","is","are","with","that","this","we","as","at","be","by","it","from"])
    freq = Counter(w for w in words if w.isalpha() and w not in stop)
    scores = []
    for p in parts:
        score = sum(freq.get(w.lower(),0) for w in p.split())
        scores.append(score)
    ranked = [p for _, p in sorted(zip(scores, parts), key=lambda t: t[0], reverse=True)]
    chosen = ranked[: min(sentences, len(ranked))]
    keywords = [w for w, _ in freq.most_common(8)]
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
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
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
