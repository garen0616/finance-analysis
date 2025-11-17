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
