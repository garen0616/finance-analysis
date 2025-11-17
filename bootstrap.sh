#!/usr/bin/env bash
set -euo pipefail

check_python() {
  python3 - <<'PY'
import sys
major, minor = sys.version_info[:2]
if major < 3 or (major == 3 and minor < 11):
    sys.exit(1)
PY
}
check_node() {
  node -v >/dev/null 2>&1 || exit 1
  node - <<'JS'
const v = process.versions.node.split('.').map(Number);
if (v[0] < 18) process.exit(1);
JS
}
if ! check_python; then
  echo "Python >=3.11 is required. Please install it and rerun." >&2
  exit 1
fi
if ! check_node; then
  echo "Node.js >=18 is required. Please install it and rerun." >&2
  exit 1
fi

mkdir -p backend/services backend/services/graph backend/storage frontend backend/external
mkdir -p backend/external backend/storage

read -r -p "Enter FMP_API_KEY (required): " FMP_API_KEY
if [ -z "${FMP_API_KEY}" ]; then
  echo "FMP_API_KEY is required." >&2
  exit 1
fi
read -r -p "Enter OPENAI_API_KEY (optional): " OPENAI_API_KEY || true
read -r -p "Enter OPENAI_SUMMARY_MODEL (optional, e.g., gpt-4o-mini): " OPENAI_SUMMARY_MODEL || true
read -r -p "Enable graph? [true]: " ENABLE_GRAPH || true
ENABLE_GRAPH=${ENABLE_GRAPH:-true}
read -r -p "Enter NEO4J_URI (optional): " NEO4J_URI || true
read -r -p "Enter NEO4J_USERNAME (optional): " NEO4J_USERNAME || true
read -r -p "Enter NEO4J_PASSWORD (optional): " NEO4J_PASSWORD || true
read -r -p "Enter NEO4J_DATABASE (optional): " NEO4J_DATABASE || true
read -r -p "Shock threshold pct [5]: " SHOCK_THRESHOLD_PCT || true
SHOCK_THRESHOLD_PCT=${SHOCK_THRESHOLD_PCT:-5}
read -r -p "Rate limit per minute [60]: " RATE_LIMIT_PER_MIN || true
RATE_LIMIT_PER_MIN=${RATE_LIMIT_PER_MIN:-60}
read -r -p "Enable FinBERT? [false]: " ENABLE_FINBERT || true
ENABLE_FINBERT=${ENABLE_FINBERT:-false}

cat > backend/.env <<EOF
FMP_API_KEY=${FMP_API_KEY}
OPENAI_API_KEY=${OPENAI_API_KEY}
OPENAI_SUMMARY_MODEL=${OPENAI_SUMMARY_MODEL}
ENABLE_GRAPH=${ENABLE_GRAPH}
NEO4J_URI=${NEO4J_URI}
NEO4J_USERNAME=${NEO4J_USERNAME}
NEO4J_PASSWORD=${NEO4J_PASSWORD}
NEO4J_DATABASE=${NEO4J_DATABASE}
SHOCK_THRESHOLD_PCT=${SHOCK_THRESHOLD_PCT}
CACHE_TTL_SECONDS=86400
RATE_LIMIT_PER_MIN=${RATE_LIMIT_PER_MIN}
ENABLE_FINBERT=${ENABLE_FINBERT}
EOF

cat > frontend/.env.local <<EOF
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
EOF

cat > .env.example <<'EOF'
FMP_API_KEY=your_fmp_premium_key
OPENAI_API_KEY=
OPENAI_SUMMARY_MODEL=gpt-4o-mini
ENABLE_GRAPH=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=neo4j
SHOCK_THRESHOLD_PCT=5
CACHE_TTL_SECONDS=86400
RATE_LIMIT_PER_MIN=60
ENABLE_FINBERT=false
EOF

cat > README.md <<'EOF'
# Earnings Analysis App

Run `bash bootstrap.sh` to scaffold, install, and launch backend (FastAPI) and frontend (Next.js 14). Backend on http://localhost:8000, frontend on http://localhost:3000.
EOF

cat > backend/requirements.txt <<'EOF'
fastapi==0.110.3
uvicorn[standard]==0.29.0
requests==2.32.3
requests-cache==1.1.1
pandas==2.2.2
numpy==1.26.4
tenacity==8.2.3
python-dotenv==1.0.1
pydantic==1.10.14
openai==1.30.1
reportlab==4.2.0
plotly==5.22.0
kaleido==0.2.1
scikit-learn==1.4.2
beautifulsoup4==4.12.3
lxml==5.2.1
neo4j==5.18.0
python-dateutil==2.9.0.post0
EOF

cat > backend/utils.py <<'EOF'
import os
import uuid
from datetime import datetime, timedelta
from dateutil import parser

def safe_div(numer, denom):
    try:
        denom = float(denom)
        return float(numer) / denom if denom else 0.0
    except Exception:
        return 0.0

def parse_date(dt):
    if not dt:
        return None
    try:
        return parser.parse(dt).date()
    except Exception:
        return None

def storage_path(base_dir, analysis_id):
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{analysis_id}.json")

def new_id():
    return str(uuid.uuid4())

def now_iso():
    return datetime.utcnow().isoformat()

def ensure_days(prices, event_date, days=20):
    if not prices:
        return []
    prices = sorted(prices, key=lambda x: x["date"])
    indexed = {p["date"]: p for p in prices}
    base_date = parse_date(event_date) or parse_date(prices[-1]["date"])
    res = []
    for i in range(-1, days + 1):
        day = base_date + timedelta(days=i)
        s = day.strftime("%Y-%m-%d")
        if s in indexed:
            res.append(indexed[s])
    return res
EOF

cat > backend/models.py <<'EOF'
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    symbol: str
    mode: str = Field("latest", regex="^(latest|specific)$")
    year: Optional[int] = None
    quarter: Optional[int] = None
    forceRefresh: bool = False
    peers: Optional[List[str]] = None

class AnalyzeResponse(BaseModel):
    analysisId: str
    summary: Dict[str, Any]
    kpis: Dict[str, Any]
    tables: Dict[str, Any]
    charts: Dict[str, str]
    transcriptHighlights: List[str]
    graphEnabled: bool

class BatchRequest(BaseModel):
    symbols: List[str]
    mode: str = Field("latest", regex="^(latest|specific)$")
    year: Optional[int] = None
    quarter: Optional[int] = None

class JobStatus(BaseModel):
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
EOF

cat > backend/services/fmp_client.py <<'EOF'
import os
from datetime import datetime, timedelta
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class FMPClient:
    def __init__(self, api_key: str, session: requests.Session | None = None):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.base = "https://financialmodelingprep.com/api/v3"
        self.base_v4 = "https://financialmodelingprep.com/api/v4"

    def _params(self, extra=None):
        p = {"apikey": self.api_key}
        if extra:
            p.update(extra)
        return p

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _get(self, url, params=None):
        r = self.session.get(url, params=self._params(params), timeout=20)
        r.raise_for_status()
        return r.json()

    def search_symbols(self, query):
        return self._get(f"{self.base}/search", {"query": query, "limit": 8})

    def get_company_profile(self, symbol):
        data = self._get(f"{self.base}/profile/{symbol}")
        return data[0] if isinstance(data, list) and data else {}

    def get_income_q(self, symbol, limit=12):
        return self._get(f"{self.base}/income-statement/{symbol}", {"period": "quarter", "limit": limit})

    def get_balance_q(self, symbol, limit=12):
        return self._get(f"{self.base}/balance-sheet-statement/{symbol}", {"period": "quarter", "limit": limit})

    def get_cashflow_q(self, symbol, limit=12):
        return self._get(f"{self.base}/cash-flow-statement/{symbol}", {"period": "quarter", "limit": limit})

    def get_earnings_company(self, symbol):
        return self._get(f"{self.base}/earnings-surprises/{symbol}")

    def get_earnings_calendar(self, symbol, from_date, to_date):
        return self._get(f"{self.base}/earning_calendar", {"from": from_date, "to": to_date, "symbol": symbol})

    def get_hist_prices(self, symbol, start_date, end_date):
        return self._get(f"{self.base}/historical-price-full/{symbol}", {"from": start_date, "to": end_date}).get("historical", [])

    def get_transcript(self, symbol, year=None, quarter=None, latest=False):
        if latest:
            try:
                all_t = self._get(f"{self.base}/earning_call_transcript/{symbol}")
                return all_t[0] if all_t else {}
            except Exception:
                return {}
        if year and quarter:
            return self._get(f"{self.base}/earning_call_transcript/{symbol}", {"year": year, "quarter": quarter})
        return {}
EOF

cat > backend/services/analysis.py <<'EOF'
import math
from datetime import datetime, timedelta
import pandas as pd
from .charts import build_charts
from .peers import compute_peer_medians
from .summarizer import summarize_text
from .rag_integration import run_agentic_pipeline
from ..utils import safe_div, ensure_days, parse_date

def _margin(numer, denom):
    return safe_div(numer, denom) * 100

def _surprise(actual, estimate):
    val = (actual or 0) - (estimate or 0)
    pct = safe_div(val, estimate or 0) * 100 if estimate else 0
    return {"value": val, "pct": pct}

def align_quarters(*frames):
    aligned = []
    for df in frames:
        aligned.append(pd.DataFrame(df))
    base = aligned[0]
    for idx, df in enumerate(aligned[1:], start=1):
        base = base.merge(df, how="outer", on=["date", "calendarYear", "period"], suffixes=("", f"_{idx}"))
    return base.sort_values(by="date" if "date" in base.columns else "calendarYear", ascending=False)

def event_windows(prices, event_date, shock_threshold):
    window_defs = {"+1": [ -1, 1 ], "+1d": [ 0, 1 ], "+5d": [ 0, 5 ], "+20d": [ 0, 20 ]}
    prices = ensure_days(prices, event_date, days=20)
    res = []
    if not prices:
        return res
    price_map = {p["date"]: p for p in prices}
    base_date = parse_date(event_date) or parse_date(prices[-1]["date"])
    for label, (start_d, end_d) in window_defs.items():
        start = base_date + timedelta(days=start_d)
        end = base_date + timedelta(days=end_d)
        start_key = start.strftime("%Y-%m-%d")
        end_key = end.strftime("%Y-%m-%d")
        if start_key in price_map and end_key in price_map:
            start_px = price_map[start_key]["adjClose"]
            end_px = price_map[end_key]["adjClose"]
            ret = safe_div(end_px - start_px, start_px) * 100
            res.append({"window": label, "returnPct": ret, "shock": abs(ret) >= shock_threshold})
    return res

def compute_kpis(income, balance, cashflow):
    res = {}
    if not income:
        return res
    latest = income[0]
    res["revenue"] = latest.get("revenue")
    res["epsDiluted"] = latest.get("epsdiluted")
    res["grossMarginPct"] = _margin(latest.get("grossProfit"), latest.get("revenue"))
    res["operatingMarginPct"] = _margin(latest.get("operatingIncome"), latest.get("revenue"))
    res["netMarginPct"] = _margin(latest.get("netIncome"), latest.get("revenue"))
    if cashflow:
        cf = cashflow[0]
        res["cfo"] = cf.get("netCashProvidedByOperatingActivities")
        res["capex"] = cf.get("capitalExpenditure")
        fcf = (cf.get("netCashProvidedByOperatingActivities") or 0) + (cf.get("capitalExpenditure") or 0)
        res["fcf"] = fcf
        rev = latest.get("revenue") or 0
        res["fcfMarginPct"] = safe_div(fcf, rev) * 100 if rev else 0
    return res

def growth_table(income):
    rows = []
    for i, row in enumerate(income[:8]):
        yoy = None
        qoq = None
        if i + 4 < len(income):
            yoy = safe_div((row.get("revenue") or 0) - (income[i+4].get("revenue") or 0), income[i+4].get("revenue") or 0) * 100
        if i + 1 < len(income):
            qoq = safe_div((row.get("revenue") or 0) - (income[i+1].get("revenue") or 0), income[i+1].get("revenue") or 0) * 100
        rows.append({"period": f"{row.get('calendarYear')}-{row.get('period')}", "revenue": row.get("revenue"), "eps": row.get("epsdiluted"), "yoyRevenuePct": yoy, "qoqRevenuePct": qoq})
    return rows

def run_analysis(symbol, mode, year, quarter, shock_threshold, client, peers_override=None):
    profile = client.get_company_profile(symbol)
    income = client.get_income_q(symbol, 12)
    balance = client.get_balance_q(symbol, 12)
    cashflow = client.get_cashflow_q(symbol, 12)
    earnings = client.get_earnings_company(symbol)
    event = earnings[0] if earnings else {}
    announcement_date = event.get("date")
    est = event.get("epsEstimated") or 0
    actual_eps = event.get("eps") or 0
    surprise = _surprise(actual_eps, est)
    cal = client.get_earnings_calendar(symbol, (datetime.utcnow() - timedelta(days=400)).strftime("%Y-%m-%d"), datetime.utcnow().strftime("%Y-%m-%d"))
    if mode == "specific" and year and quarter:
        transcript = client.get_transcript(symbol, year=year, quarter=quarter)
    else:
        transcript = client.get_transcript(symbol, latest=True)
    transcript_text = transcript.get("content") if isinstance(transcript, dict) else ""
    hist = client.get_hist_prices(symbol, (datetime.utcnow() - timedelta(days=500)).strftime("%Y-%m-%d"), datetime.utcnow().strftime("%Y-%m-%d"))
    events = event_windows(hist, announcement_date, shock_threshold)
    peers = compute_peer_medians(client, symbol, peers_override)
    charts = build_charts(income, cashflow, events, announcement_date)
    rag = run_agentic_pipeline(transcript_text or "", {"income": income, "cashflow": cashflow})
    summary = summarize_text(transcript_text or "", rag, profile)
    kpis = compute_kpis(income, balance, cashflow)
    tbls = {
        "income": income[:8],
        "balance": balance[:8],
        "cashflow": cashflow[:8],
        "yoy": growth_table(income),
        "qoq": growth_table(income),
        "eventWindows": events,
        "peers": peers,
    }
    highlights = []
    if rag and rag.get("insights"):
        highlights = [f"{i.get('text','')}" for i in rag["insights"]][:5]
    elif transcript_text:
        highlights = transcript_text.split(".")[:5]
    res = {
        "profile": profile,
        "surprise": surprise,
        "announcement": announcement_date,
    }
    return {
        "summary": summary,
        "kpis": kpis | res,
        "tables": tbls,
        "charts": charts,
        "transcript": transcript_text,
        "highlights": highlights,
        "rag": rag,
    }
EOF

cat > backend/services/rag_integration.py <<'EOF'
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
EOF

cat > backend/services/summarizer.py <<'EOF'
import os
import math
import logging
from typing import Dict, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

def _extractive_summary(text: str, sentences=6):
    if not text:
        return {"title": "No transcript", "bullets": []}
    parts = [p.strip() for p in text.split(".") if p.strip()]
    if not parts:
        return {"title": "No transcript", "bullets": []}
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(parts)
    n = min(sentences, len(parts))
    km = KMeans(n_clusters=n, n_init=5, random_state=0)
    km.fit(X)
    centers = km.cluster_centers_
    chosen = []
    for c in centers:
        idx = (X @ c).argmax()
        chosen.append(parts[idx])
    keywords = [w for w, _ in Counter(" ".join(parts).lower().split()).most_common(8)]
    return {"title": "Executive Summary", "bullets": chosen, "keywords": keywords}

def _openai_summary(text: str, model: str | None):
    client = OpenAI()
    prompt = "Summarize as a concise sell-side earnings recap with 6-10 bullets and one-sentence overview."
    resp = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text[:12000]},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    bullets = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
    return {"title": "Executive Summary", "bullets": bullets}

def summarize_text(text: str, rag: Dict[str, Any] | None, profile: Dict[str, Any]):
    openai_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_SUMMARY_MODEL") or None
    if openai_key:
        try:
            return _openai_summary(text, model)
        except Exception as exc:
            logger.warning("OpenAI summary failed: %s", exc)
    base = _extractive_summary(text)
    if rag and rag.get("insights"):
        base["bullets"] = base.get("bullets", []) + [i.get("text", "") for i in rag["insights"][:3]]
    return base
EOF

cat > backend/services/peers.py <<'EOF'
from statistics import median
from collections import defaultdict

def resolve_peers(client, symbol):
    prof = client.get_company_profile(symbol)
    peers = []
    if prof.get("companyPeers"):
        peers = [p for p in prof["companyPeers"].split(",") if p]
    elif prof.get("sector"):
        # fallback: search by sector
        peers = []
    return peers[:8]

def compute_peer_medians(client, symbol, peers_override=None):
    peers = peers_override or resolve_peers(client, symbol)
    if symbol in peers:
        peers.remove(symbol)
    metrics = defaultdict(list)
    for peer in peers[:8]:
        inc = client.get_income_q(peer, 4)
        if not inc:
            continue
        latest = inc[0]
        metrics["revenue"].append(latest.get("revenue") or 0)
        metrics["eps"].append(latest.get("epsdiluted") or 0)
        metrics["grossMargin"].append(((latest.get("grossProfit") or 0) / (latest.get("revenue") or 1)) * 100)
    medians = {k: median(v) if v else 0 for k, v in metrics.items()}
    return {"peers": peers, "medians": medians}
EOF

cat > backend/services/charts.py <<'EOF'
import base64
from io import BytesIO
import plotly.graph_objects as go

def _to_png(fig):
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return base64.b64encode(img_bytes).decode()

def build_charts(income, cashflow, event_windows, announcement_date):
    charts = {}
    if income:
        rev_fig = go.Figure()
        rev_fig.add_trace(go.Bar(x=[f"{r.get('calendarYear')}-{r.get('period')}" for r in income[:12][::-1]], y=[r.get("revenue") for r in income[:12][::-1]], name="Revenue"))
        charts["revenueTrendPng"] = f"data:image/png;base64,{_to_png(rev_fig)}"
        eps_fig = go.Figure()
        eps_fig.add_trace(go.Scatter(x=[f"{r.get('calendarYear')}-{r.get('period')}" for r in income[:12][::-1]], y=[r.get("epsdiluted") for r in income[:12][::-1]], mode="lines+markers", name="EPS"))
        charts["epsTrendPng"] = f"data:image/png;base64,{_to_png(eps_fig)}"
    if cashflow:
        fcf_fig = go.Figure()
        fcf = []
        labs = []
        for r in cashflow[:12][::-1]:
            labs.append(f"{r.get('calendarYear')}-{r.get('period')}")
            fcf.append((r.get("netCashProvidedByOperatingActivities") or 0) + (r.get("capitalExpenditure") or 0))
        fcf_fig.add_trace(go.Bar(x=labs, y=fcf, name="FCF"))
        charts["fcfTrendPng"] = f"data:image/png;base64,{_to_png(fcf_fig)}"
    if event_windows:
        ev_fig = go.Figure()
        ev_fig.add_trace(go.Bar(x=[e["window"] for e in event_windows], y=[e["returnPct"] for e in event_windows], name="Return %"))
        charts["eventStudyPng"] = f"data:image/png;base64,{_to_png(ev_fig)}"
    return charts
EOF

cat > backend/services/report_pdf.py <<'EOF'
import os
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def build_pdf(output_path, symbol, analysis, charts):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, f"Earnings Report: {symbol}")
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 96, "Executive Summary")
    y = height - 120
    for bullet in analysis["summary"].get("bullets", [])[:10]:
        c.drawString(80, y, f"- {bullet[:110]}")
        y -= 16
        if y < 100:
            c.showPage()
            y = height - 72
    c.showPage()
    for key, data_url in charts.items():
        if not data_url:
            continue
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height - 72, key)
        png_bytes = BytesIO()
        png_bytes.write(b64_to_bytes(data_url))
        png_bytes.seek(0)
        img = ImageReader(png_bytes)
        c.drawImage(img, 72, height/2 - 100, width=width-144, preserveAspectRatio=True, mask='auto')
        c.showPage()
    c.save()
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())

def b64_to_bytes(data_url):
    import base64
    if "," in data_url:
        data_url = data_url.split(",",1)[1]
    return base64.b64decode(data_url)
EOF

cat > backend/services/jobs.py <<'EOF'
import uuid
from typing import Dict

JOBS: Dict[str, dict] = {}

def create_job():
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending"}
    return job_id

def set_status(job_id, status, result=None, error=None):
    if job_id not in JOBS:
        JOBS[job_id] = {}
    JOBS[job_id]["status"] = status
    if result is not None:
        JOBS[job_id]["result"] = result
    if error is not None:
        JOBS[job_id]["error"] = error

def get_job(job_id):
    return JOBS.get(job_id, {"status": "unknown"})
EOF

cat > backend/services/limiter.py <<'EOF'
import time
from fastapi import HTTPException, status

class TokenBucket:
    def __init__(self, rate_per_min=60, capacity=None):
        self.rate = rate_per_min / 60.0
        self.capacity = capacity or rate_per_min
        self.tokens = self.capacity
        self.timestamp = time.time()

    def consume(self, amount=1):
        now = time.time()
        delta = now - self.timestamp
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        if self.tokens < amount:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
        self.tokens -= amount

buckets = {}

def check_rate_limit(ip: str, rate_per_min: int):
    bucket = buckets.get(ip)
    if not bucket:
        bucket = TokenBucket(rate_per_min=rate_per_min)
        buckets[ip] = bucket
    bucket.consume()
EOF

cat > backend/services/graph/graph_client.py <<'EOF'
import os
import logging
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class GraphClient:
    def __init__(self):
        self.enabled = os.environ.get("ENABLE_GRAPH", "true").lower() == "true"
        if not self.enabled:
            self.driver = None
            return
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USERNAME")
        pwd = os.environ.get("NEO4J_PASSWORD")
        if not (uri and user and pwd):
            self.enabled = False
            self.driver = None
            logger.warning("Graph disabled: missing credentials")
            return
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd), database=os.environ.get("NEO4J_DATABASE") or None)

    def run(self, query, params=None):
        if not self.enabled or not self.driver:
            return None
        with self.driver.session() as session:
            return session.run(query, params or {})

    def close(self):
        if self.driver:
            self.driver.close()
EOF

cat > backend/services/graph/graph_schema.cypher <<'EOF'
CREATE CONSTRAINT company_key IF NOT EXISTS FOR (c:Company) REQUIRE c.symbol IS UNIQUE;
CREATE CONSTRAINT quarter_key IF NOT EXISTS FOR (q:Quarter) REQUIRE q.id IS UNIQUE;
CREATE CONSTRAINT metric_key IF NOT EXISTS FOR (m:Metric) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT insight_key IF NOT EXISTS FOR (i:Insight) REQUIRE i.id IS UNIQUE;
EOF

cat > backend/services/graph/graph_writer.py <<'EOF'
import uuid

def ensure_schema(client):
    if not client or not client.enabled:
        return
    client.run(open("services/graph/graph_schema.cypher").read())

def write_analysis(client, symbol, analysis):
    if not client or not client.enabled:
        return
    company_id = symbol.upper()
    client.run("MERGE (c:Company {symbol:$symbol})", {"symbol": company_id})
    if analysis.get("tables", {}).get("income"):
        latest = analysis["tables"]["income"][0]
        qid = f"{company_id}-{latest.get('calendarYear')}-{latest.get('period')}"
        client.run("MERGE (q:Quarter {id:$id}) SET q.year=$year,q.period=$period", {"id": qid, "year": latest.get("calendarYear"), "period": latest.get("period")})
        client.run("MATCH (c:Company {symbol:$symbol}), (q:Quarter {id:$qid}) MERGE (c)-[:REPORTS]->(q)", {"symbol": company_id, "qid": qid})
    if analysis.get("highlights"):
        for h in analysis["highlights"][:5]:
            iid = str(uuid.uuid4())
            client.run("MERGE (i:Insight {id:$id}) SET i.text=$text", {"id": iid, "text": h})
            client.run("MATCH (c:Company {symbol:$symbol}), (i:Insight {id:$id}) MERGE (c)-[:HAS_INSIGHT]->(i)", {"symbol": company_id, "id": iid})
EOF

cat > backend/main.py <<'EOF'
import os
import json
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import requests_cache

from models import AnalyzeRequest, AnalyzeResponse, BatchRequest, JobStatus
from services.fmp_client import FMPClient
from services.analysis import run_analysis
from services.jobs import create_job, set_status, get_job
from services.limiter import check_rate_limit
from services.report_pdf import build_pdf
from services.graph.graph_client import GraphClient
from services.graph.graph_writer import ensure_schema, write_analysis
from utils import storage_path, new_id, now_iso

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("earnings-app")

CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))
requests_cache.install_cache("fmp_cache", expire_after=CACHE_TTL)
app = FastAPI(title="Earnings Analysis API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])

FMP_API_KEY = os.environ.get("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError("Missing FMP_API_KEY in environment")
client = FMPClient(FMP_API_KEY)
graph_client = GraphClient()
ensure_schema(graph_client)
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
SHOCK_THRESHOLD_PCT = float(os.environ.get("SHOCK_THRESHOLD_PCT", "5"))
GRAPH_ENABLED = os.environ.get("ENABLE_GRAPH", "true").lower() == "true"

def _analysis_path(aid):
    return storage_path(os.path.join(os.path.dirname(__file__), "storage"), aid)

def rate_limit_dependency(request: Request):
    ip = request.client.host if request.client else "local"
    check_rate_limit(ip, RATE_LIMIT)

@app.get("/api/health")
async def health():
    ok = {"fmpKey": bool(FMP_API_KEY), "cache": True, "graph": graph_client.enabled, "openai": bool(os.environ.get("OPENAI_API_KEY"))}
    return ok

@app.get("/api/search")
async def search(q: str):
    return client.search_symbols(q)

@app.get("/api/peers")
async def peers(symbol: str):
    from services.peers import compute_peer_medians
    return compute_peer_medians(client, symbol)

@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    return get_job(job_id)

async def process_analysis(req: AnalyzeRequest, aid: str):
    try:
        analysis = run_analysis(req.symbol, req.mode, req.year, req.quarter, SHOCK_THRESHOLD_PCT, client, req.peers)
        path = _analysis_path(aid)
        with open(path, "w") as f:
            json.dump(analysis, f)
        if graph_client.enabled:
            write_analysis(graph_client, req.symbol, analysis)
        set_status(aid, "completed", result={"analysisId": aid})
    except Exception as exc:
        logger.exception("Analysis error")
        set_status(aid, "failed", error=str(exc))

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest, request: Request, dep=Depends(rate_limit_dependency)):
    aid = new_id()
    analysis = run_analysis(req.symbol, req.mode, req.year, req.quarter, SHOCK_THRESHOLD_PCT, client, req.peers)
    path = _analysis_path(aid)
    with open(path, "w") as f:
        json.dump(analysis, f)
    if graph_client.enabled:
        write_analysis(graph_client, req.symbol, analysis)
    resp = AnalyzeResponse(
        analysisId=aid,
        summary=analysis["summary"],
        kpis=analysis["kpis"],
        tables=analysis["tables"],
        charts=analysis["charts"],
        transcriptHighlights=analysis["highlights"],
        graphEnabled=graph_client.enabled,
    )
    return resp

@app.post("/api/analyze-batch")
async def analyze_batch(req: BatchRequest, background: BackgroundTasks, dep=Depends(rate_limit_dependency)):
    job_id = create_job()
    async def worker():
        for sym in req.symbols:
            analysis_req = AnalyzeRequest(symbol=sym, mode=req.mode, year=req.year, quarter=req.quarter)
            await process_analysis(analysis_req, job_id)
    background.add_task(worker)
    return {"jobId": job_id}

@app.get("/api/report/pdf/{analysis_id}")
async def pdf_report(analysis_id: str):
    path = _analysis_path(analysis_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Analysis not found")
    data = json.load(open(path))
    pdf_path = os.path.join("/tmp", f"{analysis_id}.pdf")
    build_pdf(pdf_path, data.get("kpis", {}).get("profile", {}).get("symbol", analysis_id), data, data.get("charts", {}))
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"{analysis_id}.pdf")

@app.get("/api/graph/preview")
async def graph_preview(symbol: str, year: int | None = None, quarter: int | None = None):
    if not graph_client.enabled:
        return {"enabled": False, "nodes": [], "links": []}
    q = "MATCH (c:Company {symbol:$symbol})-[r]->(n) RETURN c,n,r LIMIT 50"
    data = graph_client.run(q, {"symbol": symbol})
    nodes = []
    links = []
    if data:
        for record in data:
            c = record["c"]
            n = record["n"]
            r = record["r"]
            nodes.append({"id": c.id, "label": list(c.labels)[0]})
            nodes.append({"id": n.id, "label": list(n.labels)[0]})
            links.append({"source": c.id, "target": n.id, "type": r.type})
    return {"enabled": True, "nodes": nodes, "links": links}

EOF

cat > backend/storage/.gitkeep <<'EOF'
EOF

python3 -m venv backend/.venv
backend/.venv/bin/pip install --upgrade pip
backend/.venv/bin/pip install -r backend/requirements.txt
if [ "${ENABLE_FINBERT}" = "true" ]; then
  backend/.venv/bin/pip install transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
fi

if [ ! -d "backend/external/EarningsCallAgenticRag/.git" ]; then
  git clone --depth 1 https://github.com/la9806958/EarningsCallAgenticRag backend/external/EarningsCallAgenticRag
fi

cat > frontend/package.json <<'EOF'
{
  "name": "frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "start": "next start -p 3000"
  },
  "dependencies": {
    "autoprefixer": "10.4.17",
    "lucide-react": "0.344.0",
    "next": "14.2.3",
    "plotly.js-dist-min": "2.32.0",
    "postcss": "8.4.38",
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "react-force-graph-2d": "1.24.1",
    "react-plotly.js": "2.6.0",
    "tailwindcss": "3.4.3"
  },
  "devDependencies": {
    "@types/node": "20.12.7",
    "@types/react": "18.3.2",
    "typescript": "5.4.5"
  }
}
EOF

cat > frontend/tsconfig.json <<'EOF'
{
  "compilerOptions": {
    "target": "es2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": false,
    "noEmit": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
EOF

cat > frontend/next.config.js <<'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {};
module.exports = nextConfig;
EOF

cat > frontend/tailwind.config.js <<'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui'] }
    },
  },
  plugins: [],
}
EOF

cat > frontend/postcss.config.js <<'EOF'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOF

cat > frontend/next-env.d.ts <<'EOF'
/// <reference types="next" />
/// <reference types="next/image-types/global" />
EOF

cat > frontend/app/globals.css <<'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;
:root { color-scheme: light; }
body { @apply bg-gray-50 text-gray-900; }
.card { @apply rounded-lg border bg-white shadow-sm; }
EOF

cat > frontend/app/layout.tsx <<'EOF'
import './globals.css'
import type { ReactNode } from 'react'

export const metadata = {
  title: 'Earnings Analysis',
  description: 'Earnings analytics with FMP + RAG'
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        {children}
      </body>
    </html>
  )
}
EOF

cat > frontend/app/page.tsx <<'EOF'
'use client'
import { useEffect, useMemo, useState } from 'react'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false })

type Analysis = {
  analysisId: string
  summary: { title: string; bullets: string[] }
  kpis: any
  tables: any
  charts: Record<string, string>
  transcriptHighlights: string[]
  graphEnabled: boolean
}

const backend = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export default function Page() {
  const [symbol, setSymbol] = useState('AAPL')
  const [mode, setMode] = useState<'latest'|'specific'>('latest')
  const [year, setYear] = useState<number|undefined>(undefined)
  const [quarter, setQuarter] = useState<number|undefined>(undefined)
  const [loading, setLoading] = useState(false)
  const [analysis, setAnalysis] = useState<Analysis|null>(null)
  const [symbols, setSymbols] = useState<any[]>([])
  const [query, setQuery] = useState('')
  const [graphData, setGraphData] = useState<{nodes:any[],links:any[]}>({nodes:[],links:[]})

  useEffect(() => {
    if (query.length < 1) return
    const t = setTimeout(async () => {
      const res = await fetch(`${backend}/api/search?q=${encodeURIComponent(query)}`)
      const data = await res.json()
      setSymbols(data || [])
    }, 250)
    return () => clearTimeout(t)
  }, [query])

  const generate = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${backend}/api/analyze`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({symbol, mode, year, quarter})
      })
      const data = await res.json()
      setAnalysis(data)
      if (data.graphEnabled) {
        const g = await fetch(`${backend}/api/graph/preview?symbol=${symbol}`)
        const gd = await g.json()
        setGraphData({nodes: gd.nodes || [], links: gd.links || []})
      }
    } finally {
      setLoading(false)
    }
  }

  const downloadPdf = async () => {
    if (!analysis) return
    const res = await fetch(`${backend}/api/report/pdf/${analysis.analysisId}`)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${symbol}-report.pdf`
    a.click()
    URL.revokeObjectURL(url)
  }

  const charts = useMemo(() => {
    if (!analysis?.charts) return []
    return Object.entries(analysis.charts).map(([k, v]) => ({key: k, url: v}))
  }, [analysis])

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Earnings Analysis</h1>
          <p className="text-sm text-gray-500">FMP + Agentic RAG + PDF export</p>
        </div>
        <button onClick={downloadPdf} disabled={!analysis} className="px-3 py-2 rounded bg-indigo-600 text-white disabled:opacity-50">Download PDF</button>
      </header>

      <div className="card p-4 space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="text-sm text-gray-600">Symbol</label>
            <input value={symbol} onChange={e => {setSymbol(e.target.value.toUpperCase()); setQuery(e.target.value)}} className="w-full border rounded px-3 py-2" list="symbol-options" />
            <datalist id="symbol-options">
              {symbols.map((s:any) => <option key={s.symbol} value={s.symbol}>{s.name}</option>)}
            </datalist>
          </div>
          <div>
            <label className="text-sm text-gray-600">Mode</label>
            <select value={mode} onChange={e => setMode(e.target.value as any)} className="w-full border rounded px-3 py-2">
              <option value="latest">Latest</option>
              <option value="specific">Specific</option>
            </select>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-sm text-gray-600">Year</label>
              <input type="number" value={year ?? ''} onChange={e => setYear(e.target.value ? Number(e.target.value) : undefined)} className="w-full border rounded px-3 py-2" />
            </div>
            <div>
              <label className="text-sm text-gray-600">Quarter</label>
              <input type="number" min={1} max={4} value={quarter ?? ''} onChange={e => setQuarter(e.target.value ? Number(e.target.value) : undefined)} className="w-full border rounded px-3 py-2" />
            </div>
          </div>
        </div>
        <button onClick={generate} disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded">{loading ? 'Generating...' : 'Generate'}</button>
      </div>

      {analysis && (
        <>
          <div className="card p-4 space-y-2 sticky top-0 z-10">
            <h2 className="text-xl font-semibold">{analysis.summary?.title || 'Executive Summary'}</h2>
            <ul className="list-disc pl-5 space-y-1">
              {analysis.summary?.bullets?.map((b: string, i: number) => <li key={i}>{b}</li>)}
            </ul>
          </div>

          <div className="card p-4 space-y-3">
            <h3 className="text-lg font-semibold">KPIs</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(analysis.kpis || {}).map(([k,v]) => (
                <div key={k} className="p-3 rounded border">
                  <div className="text-xs uppercase text-gray-500">{k}</div>
                  <div className="text-lg font-semibold tabular-nums">{typeof v === 'number' ? v.toFixed(2) : String(v)}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {charts.map(c => (
              <div key={c.key} className="card p-2">
                <div className="text-sm font-semibold px-2 py-1">{c.key}</div>
                <img src={c.url} alt={c.key} className="w-full" />
              </div>
            ))}
          </div>

          <div className="card p-4">
            <h3 className="text-lg font-semibold mb-2">Transcript Highlights</h3>
            <ul className="list-disc pl-5 space-y-1">
              {analysis.transcriptHighlights.map((h, i) => <li key={i}>{h}</li>)}
            </ul>
          </div>

          {analysis.graphEnabled && (
            <div className="card p-4">
              <h3 className="text-lg font-semibold mb-2">Graph Preview</h3>
              <div className="h-80">
                <ForceGraph2D
                  graphData={graphData}
                  nodeLabel={(n:any)=>n.id}
                  nodeAutoColorBy="label"
                />
              </div>
            </div>
          )}

          <div className="card p-4">
            <h3 className="text-lg font-semibold mb-2">Raw Data</h3>
            <pre className="text-xs overflow-auto max-h-96">{JSON.stringify(analysis.tables, null, 2)}</pre>
          </div>
        </>
      )}
    </div>
  )
}
EOF

(cd frontend && npm install)

(cd backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000 > ../backend.log 2>&1 &) 

echo "Starting frontend..."
(cd frontend && npm run dev)
EOF
chmod +x bootstrap.sh
