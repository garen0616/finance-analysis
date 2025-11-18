import os
import json
import logging
import typing

# Python 3.13 compatibility for pydantic v1 ForwardRef._evaluate signature
if hasattr(typing, "ForwardRef"):
    _fr_eval = typing.ForwardRef._evaluate
    def _patched_forward_eval(self, globalns=None, localns=None, type_params=None, recursive_guard=None):
        if recursive_guard is None:
            recursive_guard = set()
        try:
            return _fr_eval(self, globalns, localns, type_params, recursive_guard=recursive_guard)
        except TypeError:
            # Fallback for older signature
            return _fr_eval(self, globalns, localns, recursive_guard=recursive_guard)
        except Exception:
            return _fr_eval(self, globalns, localns, type_params, recursive_guard)
    typing.ForwardRef._evaluate = _patched_forward_eval

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
