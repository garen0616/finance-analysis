#!/usr/bin/env bash
set -euo pipefail

# locate dirs
if [ -d "repo/backend" ] && [ -d "repo/frontend" ]; then
  BACKEND_DIR="repo/backend"
  FRONTEND_DIR="repo/frontend"
elif [ -d "backend" ] && [ -d "frontend" ]; then
  BACKEND_DIR="backend"
  FRONTEND_DIR="frontend"
else
  echo "backend/frontend not found" >&2
  exit 1
fi

# ensure DATA_ROOT in backend/.env
if [ -f "$BACKEND_DIR/.env" ] && ! grep -q "^DATA_ROOT=" "$BACKEND_DIR/.env"; then
  echo "DATA_ROOT=backend/external/EarningsCallAgenticRag" >> "$BACKEND_DIR/.env"
fi

# ensure deps in requirements
if ! grep -q "python-dateutil" "$BACKEND_DIR/requirements.txt"; then echo "python-dateutil==2.9.0.post0" >> "$BACKEND_DIR/requirements.txt"; fi
if ! grep -q "pyarrow" "$BACKEND_DIR/requirements.txt"; then echo "pyarrow==16.1.0" >> "$BACKEND_DIR/requirements.txt"; fi

cat > "$BACKEND_DIR/services/repo_data.py" <<'PYEOF'
import os
import glob
import pandas as pd
from typing import List, Dict, Optional
from dateutil import parser

DATA_ROOT = os.environ.get("DATA_ROOT", "backend/external/EarningsCallAgenticRag")

TRANSCRIPT_FILES = ["maec_transcripts.csv", "merged_data_nasdaq.csv", "merged_data_nyse.csv"]
TRANSCRIPT_COLS = ["transcript", "text", "content", "call_text"]
DATE_COLS = ["date", "earnings_date", "announce_date", "call_date"]
TICKER_COLS = ["symbol", "ticker"]
YEAR_COLS = ["fiscal_year", "fiscalYear", "year"]
QTR_COLS = ["fiscal_quarter", "fiscalQuarter", "quarter", "fiscalPeriod"]

class RepoDataLoader:
    def __init__(self, root: str = DATA_ROOT):
        self.root = root

    def _datasets(self):
        ds = []
        for fn in TRANSCRIPT_FILES:
            path = os.path.join(self.root, fn)
            if os.path.isfile(path):
                ds.append(path)
        return ds

    def list_datasets(self) -> List[Dict]:
        out = []
        for path in self._datasets():
            out.append({"name": os.path.basename(path), "path": path, "size": os.path.getsize(path)})
        return out

    def _read(self, path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _find_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        for c in candidates:
            low = c.lower()
            for col in df.columns:
                if col.lower() == low:
                    return col
        return None

    def _get_dataset_path(self, dataset: str) -> str:
        if os.path.isabs(dataset):
            return dataset
        for path in self._datasets():
            if os.path.basename(path) == dataset:
                return path
        raise FileNotFoundError(f"dataset not found: {dataset}")

    def list_tickers(self, dataset: str) -> List[Dict]:
        path = self._get_dataset_path(dataset)
        df = self._read(path)
        tcol = self._find_col(df, TICKER_COLS)
        if not tcol:
            return []
        counts = df[tcol].value_counts().to_dict()
        return [{"symbol": k, "count": int(v)} for k, v in counts.items()]

    def list_periods(self, dataset: str, symbol: str) -> List[Dict]:
        path = self._get_dataset_path(dataset)
        df = self._read(path)
        tcol = self._find_col(df, TICKER_COLS)
        if not tcol:
            return []
        df = df[df[tcol].str.upper() == symbol.upper()]
        ycol = self._find_col(df, YEAR_COLS)
        qcol = self._find_col(df, QTR_COLS)
        dcol = self._find_col(df, DATE_COLS)
        periods = []
        for _, row in df.iterrows():
            fy = int(row[ycol]) if ycol in row and not pd.isna(row[ycol]) else None
            fq = None
            if qcol in row and not pd.isna(row[qcol]):
                try:
                    fq = int(str(row[qcol]).replace("Q", "").replace("q", ""))
                except Exception:
                    fq = None
            dt = None
            if dcol in row and not pd.isna(row[dcol]):
                try:
                    dt = parser.parse(str(row[dcol])).date().isoformat()
                except Exception:
                    dt = None
            periods.append({"fiscalYear": fy, "fiscalQuarter": fq, "periodEnd": dt})
        return periods

    def load_transcript(self, dataset: str, symbol: str, year: Optional[int], quarter: Optional[int]):
        path = self._get_dataset_path(dataset)
        df = self._read(path)
        tcol = self._find_col(df, TICKER_COLS)
        if not tcol:
            return None
        df = df[df[tcol].str.upper() == symbol.upper()]
        ycol = self._find_col(df, YEAR_COLS)
        qcol = self._find_col(df, QTR_COLS)
        if year and ycol in df.columns:
            df = df[df[ycol] == year]
        if quarter and qcol in df.columns:
            try:
                df = df[df[qcol].astype(str).str.replace("Q","", regex=False).astype(int) == int(quarter)]
            except Exception:
                df = df
        if df.empty:
            return None
        tfield = self._find_col(df, TRANSCRIPT_COLS)
        dfield = self._find_col(df, DATE_COLS)
        row = df.iloc[0]
        text = str(row[tfield]) if tfield in row and not pd.isna(row[tfield]) else ""
        date = None
        if dfield in row and not pd.isna(row[dfield]):
            try:
                date = parser.parse(str(row[dfield])).date().isoformat()
            except Exception:
                date = None
        return {"text": text, "date": date}

    def load_statements(self, symbol: str, year: Optional[int], quarter: Optional[int]):
        fin_dir = os.path.join(self.root, "financial_statements")
        if not os.path.isdir(fin_dir):
            return None
        def find_file(kind: str):
            pat = os.path.join(fin_dir, f"{symbol.upper()}_{kind}*.csv")
            matches = glob.glob(pat)
            if matches:
                return matches[0]
            return None
        income_p = find_file("income")
        bal_p = find_file("balance")
        cf_p = find_file("cashflow")
        if not income_p and not bal_p and not cf_p:
            return None

        def load_filtered(path: Optional[str]):
            if not path: return []
            df = self._read(path)
            ycol = self._find_col(df, YEAR_COLS)
            qcol = self._find_col(df, QTR_COLS)
            if year and ycol in df.columns:
                df = df[df[ycol] == year]
            if quarter and qcol in df.columns:
                try:
                    df = df[df[qcol].astype(str).str.replace("Q","", regex=False).astype(int) == int(quarter)]
                except Exception:
                    df = df
            return df.to_dict(orient="records")
        return {
            "income_df": load_filtered(income_p),
            "balance_df": load_filtered(bal_p),
            "cashflow_df": load_filtered(cf_p),
        }
PYEOF

cat > "$BACKEND_DIR/models.py" <<'PYEOF'
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    source: str = Field("fmp", regex="^(fmp|repo)$")
    dataset: Optional[str] = None
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
PYEOF

cat > "$BACKEND_DIR/services/analysis.py" <<'PYEOF'
import math
from datetime import datetime, timedelta
import pandas as pd
from fastapi import HTTPException
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

def compute_kpis(income, balance, cashflow):
    res = {}
    if not income:
        return res
    latest = income[0]
    res["revenue"] = latest.get("revenue")
    res["epsDiluted"] = latest.get("epsdiluted") or latest.get("epsDiluted")
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
        rows.append({"period": f"{row.get('calendarYear') or row.get('fiscalYear')}-{row.get('period') or row.get('fiscalPeriod')}", "revenue": row.get("revenue"), "eps": row.get("epsdiluted") or row.get("epsDiluted"), "yoyRevenuePct": yoy, "qoqRevenuePct": qoq})
    return rows

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
            start_px = price_map[start_key].get("adjClose") or price_map[start_key].get("close")
            end_px = price_map[end_key].get("adjClose") or price_map[end_key].get("close")
            ret = safe_div(end_px - start_px, start_px) * 100
            res.append({"window": label, "returnPct": ret, "shock": abs(ret) >= shock_threshold})
    return res

def run_analysis(symbol, mode, year, quarter, shock_threshold, client, peers_override=None, source="fmp", dataset=None, repo_loader=None):
    if source == "repo":
        if not repo_loader:
            raise HTTPException(status_code=400, detail="Repo loader unavailable")
        transcript = repo_loader.load_transcript(dataset, symbol, year, quarter) if dataset else None
        transcript_text = (transcript or {}).get("text") or ""
        transcript_date = (transcript or {}).get("date")
        stmts = repo_loader.load_statements(symbol, year, quarter)
        if not stmts:
            raise HTTPException(status_code=400, detail="Statements not found in repo for symbol; try FMP Live.")
        income = stmts.get("income_df") or []
        balance = stmts.get("balance_df") or []
        cashflow = stmts.get("cashflow_df") or []
        kpis = compute_kpis(income, balance, cashflow)
        charts = build_charts(income, cashflow, [], transcript_date)
        hist = client.get_hist_prices(symbol, (datetime.utcnow() - timedelta(days=500)).strftime("%Y-%m-%d"), datetime.utcnow().strftime("%Y-%m-%d")) if transcript_date else []
        events = event_windows(hist, transcript_date, shock_threshold) if transcript_date else []
        rag = run_agentic_pipeline(transcript_text or "", {"income": income, "cashflow": cashflow})
        summary = summarize_text(transcript_text or "", rag, {})
        tbls = {
            "income": income[:8],
            "balance": balance[:8],
            "cashflow": cashflow[:8],
            "yoy": growth_table(income),
            "qoq": growth_table(income),
            "eventWindows": events,
            "peers": {"peers": [], "medians": {}},
        }
        highlights = []
        if rag and rag.get("insights"):
            highlights = [f"{i.get('text','')}" for i in rag["insights"]][:5]
        elif transcript_text:
            highlights = transcript_text.split(".")[:5]
        res = {
            "summary": summary,
            "kpis": kpis | {"profile": {"symbol": symbol}},
            "tables": tbls,
            "charts": charts,
            "transcript": transcript_text,
            "highlights": highlights,
            "rag": rag,
            "announcement": transcript_date,
        }
        return res

    profile = client.get_company_profile(symbol)
    income = client.get_income_q(symbol, 12)
    balance = client.get_balance_q(symbol, 12)
    cashflow = client.get_cashflow_q(symbol, 12)
    earnings = client.get_earnings_company(symbol)
    event = earnings[0] if earnings else {}
    announcement_date = event.get("date")
    est = event.get("epsEstimated") or event.get("epsEstimate") or event.get("epsestimated") or 0
    actual_eps = event.get("epsActual") or event.get("eps") or 0
    surprise = _surprise(actual_eps, est)
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
PYEOF

cat > "$BACKEND_DIR/main.py" <<'PYEOF'
import os
import json
import logging
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
from services.repo_data import RepoDataLoader
from utils import storage_path, new_id, now_iso

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("earnings-app")

CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))
requests_cache.install_cache("fmp_cache", expire_after=CACHE_TTL)
app = FastAPI(title="Earnings Analysis API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FMP_API_KEY = os.environ.get("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError("Missing FMP_API_KEY in environment")
client = FMPClient(FMP_API_KEY)
graph_client = GraphClient()
ensure_schema(graph_client)
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
SHOCK_THRESHOLD_PCT = float(os.environ.get("SHOCK_THRESHOLD_PCT", "5"))
repo_loader = RepoDataLoader()

def _analysis_path(aid):
    return storage_path(os.path.join(os.path.dirname(__file__), "storage"), aid)

def rate_limit_dependency(request: Request):
    ip = request.client.host if request.client else "local"
    check_rate_limit(ip, RATE_LIMIT)

@app.get("/api/health")
async def health():
    ok = {"fmpKey": bool(FMP_API_KEY), "cache": True, "graph": graph_client.enabled, "openai": bool(os.environ.get("OPENAI_API_KEY"))}
    return ok

@app.get("/api/datasources")
async def datasources():
    return {"sources": ["repo", "fmp"], "datasets": repo_loader.list_datasets()}

@app.get("/api/dataset/tickers")
async def dataset_tickers(dataset: str):
    return repo_loader.list_tickers(dataset)

@app.get("/api/dataset/periods")
async def dataset_periods(dataset: str, symbol: str):
    return repo_loader.list_periods(dataset, symbol)

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
      analysis = run_analysis(req.symbol, req.mode, req.year, req.quarter, SHOCK_THRESHOLD_PCT, client, req.peers, req.source, req.dataset, repo_loader)
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
    analysis = run_analysis(req.symbol, req.mode, req.year, req.quarter, SHOCK_THRESHOLD_PCT, client, req.peers, req.source, req.dataset, repo_loader)
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
        seen_nodes = set()
        seen_links = set()
        for record in data:
            c = record["c"]
            n = record["n"]
            r = record["r"]
            for node in [(c.id, list(c.labels)[0]), (n.id, list(n.labels)[0])]:
                if node[0] not in seen_nodes:
                    nodes.append({"id": node[0], "label": node[1]})
                    seen_nodes.add(node[0])
            link_key = (c.id, n.id, r.type)
            if link_key not in seen_links:
                links.append({"source": c.id, "target": n.id, "type": r.type})
                seen_links.add(link_key)
    return {"enabled": True, "nodes": nodes, "links": links}
PYEOF

# frontend controls
cat > "$FRONTEND_DIR/components/controls/DataSourceToggle.tsx" <<'EOF'
"use client";
import { cn } from "../ui/cn";
export default function DataSourceToggle({ value, onChange }: { value: "repo" | "fmp"; onChange: (v: "repo" | "fmp") => void }) {
  const opts = [
    { value: "repo", label: "GitHub Sample" },
    { value: "fmp", label: "FMP Live" },
  ];
  return (
    <div>
      <div className="text-xs uppercase text-slate-500">Data Source</div>
      <div className="inline-flex rounded-md border border-[var(--line)] overflow-hidden mt-1">
        {opts.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value as any)}
            className={cn("px-3 py-2 text-sm", value === opt.value ? "bg-surface-muted font-medium" : "bg-white/70")}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
