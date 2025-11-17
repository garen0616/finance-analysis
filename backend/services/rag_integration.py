import importlib.util
import os
import sys
import logging

logger = logging.getLogger(__name__)

def _load_repo():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "EarningsCallAgenticRag"))
    if not os.path.isdir(path):
        return None
    if path not in sys.path:
        sys.path.append(path)
    try:
        spec = importlib.util.find_spec("main")
        if spec:
            return importlib.util.module_from_spec(spec)
    except Exception as exc:
        logger.warning("RAG load failed: %s", exc)
    return None

def run_agentic_pipeline(transcript_text: str, financials: dict):
    rag_mod = _load_repo()
    if not rag_mod:
        return None
    try:
        if hasattr(rag_mod, "run"):
            return rag_mod.run(transcript_text, financials)
    except Exception as exc:
        logger.warning("RAG pipeline error: %s", exc)
    return None
