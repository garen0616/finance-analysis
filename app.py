import os
import sys
import math
import time
import textwrap
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer
import requests
import pandas as pd
from dateutil import parser as date_parser
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt, Confirm
from rich.text import Text


APP_NAME = "Finance Analysis CLI"
console = Console()
app = typer.Typer(add_completion=False, help=APP_NAME)


# ---------------------------
# Helpers & Formatting
# ---------------------------


def format_currency(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    try:
        return f"{value:,.0f}"
    except Exception:
        return str(value)


def format_currency_compact(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    abs_val = abs(value)
    suffix = ""
    divisor = 1.0
    if abs_val >= 1_000_000_000_000:
        suffix = "T"
        divisor = 1_000_000_000_000
    elif abs_val >= 1_000_000_000:
        suffix = "B"
        divisor = 1_000_000_000
    elif abs_val >= 1_000_000:
        suffix = "M"
        divisor = 1_000_000
    scaled = value / divisor
    return f"{scaled:,.2f}{suffix}"


def format_pct(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    try:
        return f"{value:.2f}%"
    except Exception:
        return str(value)


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    try:
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    except Exception:
        return None


def ensure_reports_dir() -> Path:
    path = Path("reports")
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_quarter_from_date(date_str: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        dt = date_parser.parse(date_str)
        quarter = (dt.month - 1) // 3 + 1
        return dt.year, quarter
    except Exception:
        return None, None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------
# FMP Client
# ---------------------------


class FMPError(RuntimeError):
    pass


class FMPClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://financialmodelingprep.com/stable",
        max_retries: int = 5,
        backoff_factor: float = 0.7,
        timeout: int = 15,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self.base_url}{path}"
        params = dict(params or {})
        params["apikey"] = self.api_key

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise FMPError(f"Network error calling FMP: {exc}") from exc
                sleep_s = self.backoff_factor * (2 ** attempt)
                time.sleep(sleep_s)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise FMPError(f"Failed to decode JSON from FMP: {exc}") from exc

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == self.max_retries - 1:
                    raise FMPError(f"FMP API error {resp.status_code}: {resp.text[:200]}")
                sleep_s = self.backoff_factor * (2 ** attempt)
                time.sleep(sleep_s)
                continue

            raise FMPError(f"FMP API error {resp.status_code}: {resp.text[:200]}")

        raise FMPError("Unexpected error contacting FMP API")

    # --- High level endpoints ---

    def search_name(self, query: str, limit: int = 10) -> pd.DataFrame:
        data = self._request("/search-name", {"query": query, "limit": limit})
        df = pd.DataFrame(data or [])
        return df

    def profile(self, symbol: str) -> pd.DataFrame:
        data = self._request("/profile", {"symbol": symbol})
        return pd.DataFrame(data or [])

    def transcript_dates(self, symbol: str) -> pd.DataFrame:
        data = self._request("/earning-call-transcript-dates", {"symbol": symbol})
        df = pd.DataFrame(data or [])
        # Normalize columns commonly present: year, quarter, date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def transcript(self, symbol: str, year: int, quarter: int) -> pd.DataFrame:
        data = self._request(
            "/earning-call-transcript",
            {"symbol": symbol, "year": year, "quarter": quarter},
        )
        return pd.DataFrame(data or [])

    def earnings(self, symbol: str) -> pd.DataFrame:
        data = self._request("/earnings", {"symbol": symbol})
        df = pd.DataFrame(data or [])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Try to infer fiscal year/quarter from available fields
        if "fiscalDateEnding" in df.columns:
            dates = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
        else:
            dates = df.get("date")
        if dates is not None:
            df["fy"] = dates.dt.year
            df["fq"] = ((dates.dt.month - 1) // 3 + 1).astype("Int64")
        return df

    def income_statement(self, symbol: str, limit: int = 8) -> pd.DataFrame:
        data = self._request("/income-statement", {"symbol": symbol, "limit": limit})
        df = pd.DataFrame(data or [])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "calendarYear" in df.columns:
            df["fy"] = pd.to_numeric(df["calendarYear"], errors="coerce").astype("Int64")
        else:
            df["fy"] = df["date"].dt.year
        if "period" in df.columns:
            df["fq"] = df["period"].astype(str).str.upper().str.replace("Q", "", regex=False)
            df["fq"] = pd.to_numeric(df["fq"], errors="coerce").astype("Int64")
        else:
            df["fq"] = ((df["date"].dt.month - 1) // 3 + 1).astype("Int64")
        return df

    def balance_sheet(self, symbol: str, limit: int = 8) -> pd.DataFrame:
        data = self._request("/balance-sheet-statement", {"symbol": symbol, "limit": limit})
        df = pd.DataFrame(data or [])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["fy"] = df["date"].dt.year
            df["fq"] = ((df["date"].dt.month - 1) // 3 + 1).astype("Int64")
        return df

    def cashflow(self, symbol: str, limit: int = 8) -> pd.DataFrame:
        data = self._request("/cashflow-statement", {"symbol": symbol, "limit": limit})
        df = pd.DataFrame(data or [])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["fy"] = df["date"].dt.year
            df["fq"] = ((df["date"].dt.month - 1) // 3 + 1).astype("Int64")
        return df

    def historical_eod(self, symbol: str) -> pd.DataFrame:
        data = self._request("/historical-price-eod/full", {"symbol": symbol})
        df = pd.DataFrame(data or [])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
        return df


# ---------------------------
# Data models
# ---------------------------


@dataclass
class QuarterSelection:
    symbol: str
    year: int
    quarter: int


@dataclass
class EarningsHighlight:
    announcement_date: Optional[datetime]
    time_of_day: Optional[str]
    eps_actual: Optional[float]
    eps_estimate: Optional[float]
    eps_surprise: Optional[float]
    eps_surprise_pct: Optional[float]
    revenue_actual: Optional[float]
    revenue_estimate: Optional[float]
    revenue_surprise_pct: Optional[float]


@dataclass
class EventStudyResult:
    baseline_date: Optional[datetime]
    baseline_price: Optional[float]
    event_date: Optional[datetime]
    returns: Dict[str, Optional[float]]


# ---------------------------
# Core analysis pipeline
# ---------------------------


def build_analysis(
    client: "FMPClient",
    selection: "QuarterSelection",
    lang: str = "en",
) -> Dict[str, Any]:
    symbol = selection.symbol.upper()
    year = selection.year
    quarter = selection.quarter

    profile_df = client.profile(symbol)
    income_df = client.income_statement(symbol)
    balance_df = client.balance_sheet(symbol)
    cashflow_df = client.cashflow(symbol)
    earnings_df = client.earnings(symbol)
    hist_df = client.historical_eod(symbol)

    # Income statement rows
    income_current_row = pick_quarter_row(income_df, year, quarter)
    income_yoy_row = pick_quarter_row(income_df, year - 1, quarter)
    prev_year, prev_quarter = (year - 1, 4) if quarter == 1 else (year, quarter - 1)
    income_qoq_row = pick_quarter_row(income_df, prev_year, prev_quarter)

    income_current = (
        extract_income_metrics(income_current_row)
        if income_current_row is not None
        else extract_income_metrics(None)
    )
    income_yoy = (
        extract_income_metrics(income_yoy_row)
        if income_yoy_row is not None
        else extract_income_metrics(None)
    )
    income_qoq = (
        extract_income_metrics(income_qoq_row)
        if income_qoq_row is not None
        else extract_income_metrics(None)
    )

    # Cash flow rows
    cf_current_row = pick_quarter_row(cashflow_df, year, quarter)
    cf_yoy_row = pick_quarter_row(cashflow_df, year - 1, quarter)
    cf_current = (
        extract_cashflow_metrics(cf_current_row)
        if cf_current_row is not None
        else extract_cashflow_metrics(None)
    )
    cf_yoy = (
        extract_cashflow_metrics(cf_yoy_row)
        if cf_yoy_row is not None
        else extract_cashflow_metrics(None)
    )

    # Balance sheet trend for last 4 quarters
    balance_trend_df = extract_balance_trend(balance_df)
    balance_trend: Optional[Dict[str, Any]] = None
    if not balance_trend_df.empty:
        labels = balance_trend_df["date"].dt.strftime("%Y-%m-%d").tolist()
        cols_map = {
            "cashAndShortTermInvestments": "Cash+ST Inv",
            "cashAndCashEquivalents": "Cash & Equiv",
            "shortTermInvestments": "Short-term Inv",
            "totalDebt": "Total Debt",
            "netDebt": "Net Debt",
            "inventory": "Inventory",
            "netReceivables": "Receivables",
            "totalStockholdersEquity": "Equity",
        }
        rows: List[Dict[str, Any]] = []
        for col, label in cols_map.items():
            if col not in balance_trend_df.columns:
                continue
            values = [balance_trend_df.iloc[i][col] for i in range(len(balance_trend_df))]
            rows.append({"label": label, "values": values})
        balance_trend = {"labels": labels, "rows": rows}

    # Earnings and event study
    earnings_row = select_earnings_row(earnings_df, year, quarter)
    earnings_highlight = compute_earnings_highlight(earnings_row)
    event_result = compute_event_study(hist_df, earnings_highlight)

    # Transcript
    try:
        transcript_df = client.transcript(symbol, year, quarter)
    except FMPError:
        transcript_df = pd.DataFrame()
    transcript_summary = build_transcript_summary(transcript_df, lang=lang)

    # Company info for web templates
    company: Optional[Dict[str, Any]] = None
    if not profile_df.empty:
        row = profile_df.iloc[0]
        name = row.get("companyName") or row.get("company") or row.get("name") or symbol
        exchange = row.get("exchangeShortName") or row.get("exchange") or "N/A"
        sector = row.get("sector") or "N/A"
        industry = row.get("industry") or "N/A"
        market_cap = row.get("mktCap") or row.get("marketCap")
        fiscal_year_end = row.get("fiscalYear") or row.get("fiscalYearEnd") or row.get("fiscalYearEndDate")
        company = {
            "name": name,
            "symbol": symbol,
            "exchange": exchange,
            "sector": sector,
            "industry": industry,
            "market_cap": float(market_cap) if pd.notna(market_cap) else None,
            "fiscal_year_end": fiscal_year_end if pd.notna(fiscal_year_end) else None,
        }

    return {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "lang": lang,
        "profile_df": profile_df,
        "company": company,
        "income_current": income_current,
        "income_yoy": income_yoy,
        "income_qoq": income_qoq,
        "cf_current": cf_current,
        "cf_yoy": cf_yoy,
        "balance_df": balance_df,
        "balance_trend": balance_trend,
        "earnings": earnings_highlight,
        "event": event_result,
        "transcript_summary": transcript_summary,
        "hist_df": hist_df,
    }


# ---------------------------
# Financial Calculations
# ---------------------------


def pick_quarter_row(df: pd.DataFrame, year: int, quarter: int) -> Optional[pd.Series]:
    if df.empty:
        return None
    mask = (df.get("fy") == year) & (df.get("fq") == quarter)
    filtered = df[mask]
    if not filtered.empty:
        # Pick the most recent by date
        if "date" in filtered.columns:
            filtered = filtered.sort_values("date")
        return filtered.iloc[-1]
    # Fallback: pick closest by date to the target fiscal year/quarter end
    target_date = datetime(year, quarter * 3, 1) + timedelta(days=30)
    if "date" in df.columns:
        df_non_null = df.dropna(subset=["date"]).copy()
        if df_non_null.empty:
            return None
        df_non_null["date_diff"] = (df_non_null["date"] - target_date).abs()
        df_non_null = df_non_null.sort_values("date_diff")
        return df_non_null.iloc[0]
    return None


def compute_margin(value: Optional[float], revenue: Optional[float]) -> Optional[float]:
    ratio = safe_div(value, revenue)
    if ratio is None:
        return None
    return ratio * 100.0


def compute_growth(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    ratio = safe_div(current, previous)
    if ratio is None:
        return None
    return (ratio - 1.0) * 100.0


def extract_income_metrics(row: pd.Series) -> Dict[str, Optional[float]]:
    if row is None:
        return {k: None for k in ["revenue", "gross_profit", "operating_income", "net_income"]}
    return {
        "revenue": float(row.get("revenue")) if pd.notna(row.get("revenue")) else None,
        "gross_profit": float(row.get("grossProfit")) if pd.notna(row.get("grossProfit")) else None,
        "operating_income": float(row.get("operatingIncome")) if pd.notna(row.get("operatingIncome")) else None,
        "net_income": float(row.get("netIncome")) if pd.notna(row.get("netIncome")) else None,
    }


def extract_cashflow_metrics(row: pd.Series) -> Dict[str, Optional[float]]:
    if row is None:
        return {k: None for k in ["cfo", "capex", "fcf"]}
    cfo_candidates = [
        "operatingCashFlow",
        "netCashProvidedByOperatingActivities",
        "netCashProvidedByOperatingActivitiesContinuingOperations",
    ]
    capex_candidates = ["capitalExpenditure", "capitalExpenditures", "investmentsInPropertyPlantAndEquipment"]

    def pick(field_names: Iterable[str]) -> Optional[float]:
        for name in field_names:
            if name in row and pd.notna(row.get(name)):
                try:
                    return float(row.get(name))
                except Exception:
                    continue
        return None

    cfo = pick(cfo_candidates)
    capex = pick(capex_candidates)
    fcf = cfo - capex if cfo is not None and capex is not None else None

    return {"cfo": cfo, "capex": capex, "fcf": fcf}


def extract_balance_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "date",
        "cashAndCashEquivalents",
        "shortTermInvestments",
        "cashAndShortTermInvestments",
        "totalDebt",
        "netDebt",
        "inventory",
        "netReceivables",
        "totalStockholdersEquity",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    trend = df[existing_cols].copy()
    trend = trend.sort_values("date").tail(4)
    return trend


def select_earnings_row(earnings_df: pd.DataFrame, year: int, quarter: int) -> Optional[pd.Series]:
    if earnings_df.empty:
        return None
    mask = (earnings_df.get("fy") == year) & (earnings_df.get("fq") == quarter)
    filtered = earnings_df[mask]
    if not filtered.empty:
        filtered = filtered.sort_values("date")
        return filtered.iloc[-1]
    # fallback: nearest by fiscal date
    date_col = "fiscalDateEnding" if "fiscalDateEnding" in earnings_df.columns else "date"
    if date_col in earnings_df.columns:
        df = earnings_df.dropna(subset=[date_col]).copy()
        if df.empty:
            return None
        target_date = datetime(year, quarter * 3, 1) + timedelta(days=30)
        df["date_tmp"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["date_tmp"])
        if df.empty:
            return None
        df["date_diff"] = (df["date_tmp"] - target_date).abs()
        df = df.sort_values("date_diff")
        return df.iloc[0]
    return None


def compute_earnings_highlight(row: Optional[pd.Series]) -> EarningsHighlight:
    if row is None:
        return EarningsHighlight(
            announcement_date=None,
            time_of_day=None,
            eps_actual=None,
            eps_estimate=None,
            eps_surprise=None,
            eps_surprise_pct=None,
            revenue_actual=None,
            revenue_estimate=None,
            revenue_surprise_pct=None,
        )
    date_val = row.get("date")
    if isinstance(date_val, pd.Timestamp):
        announcement_date = date_val.to_pydatetime()
    else:
        try:
            announcement_date = date_parser.parse(str(date_val)) if pd.notna(date_val) else None
        except Exception:
            announcement_date = None

    time_of_day = str(row.get("time")).upper() if row.get("time") is not None else None

    eps_actual = None
    for key in ("epsActual", "eps", "epsReported"):
        if key in row and pd.notna(row.get(key)):
            eps_actual = float(row.get(key))
            break

    eps_estimate = None
    for key in ("epsEstimated", "epsEstimate"):
        if key in row and pd.notna(row.get(key)):
            eps_estimate = float(row.get(key))
            break

    revenue_actual = None
    for key in ("revenue", "revenueActual"):
        if key in row and pd.notna(row.get(key)):
            revenue_actual = float(row.get(key))
            break

    revenue_estimate = None
    for key in ("revenueEstimated", "revenueEstimate"):
        if key in row and pd.notna(row.get(key)):
            revenue_estimate = float(row.get(key))
            break

    eps_surprise = None
    eps_surprise_pct = None
    if eps_actual is not None and eps_estimate is not None:
        eps_surprise = eps_actual - eps_estimate
        if eps_estimate != 0:
            eps_surprise_pct = (eps_surprise / abs(eps_estimate)) * 100.0

    revenue_surprise_pct = None
    if revenue_actual is not None and revenue_estimate is not None and revenue_estimate != 0:
        revenue_surprise_pct = ((revenue_actual - revenue_estimate) / abs(revenue_estimate)) * 100.0

    return EarningsHighlight(
        announcement_date=announcement_date,
        time_of_day=time_of_day,
        eps_actual=eps_actual,
        eps_estimate=eps_estimate,
        eps_surprise=eps_surprise,
        eps_surprise_pct=eps_surprise_pct,
        revenue_actual=revenue_actual,
        revenue_estimate=revenue_estimate,
        revenue_surprise_pct=revenue_surprise_pct,
    )


def compute_event_study(
    hist_df: pd.DataFrame,
    earnings: EarningsHighlight,
) -> EventStudyResult:
    if hist_df.empty or earnings.announcement_date is None:
        return EventStudyResult(
            baseline_date=None,
            baseline_price=None,
            event_date=None,
            returns={"T+1": None, "T+3": None, "T+7": None},
        )

    df = hist_df.dropna(subset=["date", "close"]).copy()
    if df.empty:
        return EventStudyResult(
            baseline_date=None,
            baseline_price=None,
            event_date=None,
            returns={"T+1": None, "T+3": None, "T+7": None},
        )

    df = df.sort_values("date").reset_index(drop=True)

    # Find index for announcement date and previous / next trading days
    ann_date = earnings.announcement_date.date()

    idx_ann = df.index[df["date"].dt.date == ann_date]
    if len(idx_ann) == 0:
        # pick closest date
        df["date_diff"] = (df["date"].dt.date - ann_date).abs()
        idx_ann = [df["date_diff"].idxmin()]
    idx_ann = int(idx_ann[0])

    idx_prev = max(0, idx_ann - 1)
    baseline_row = df.iloc[idx_prev]
    baseline_date = baseline_row["date"].to_pydatetime()
    baseline_price = float(baseline_row.get("close"))

    # Determine event index based on BMO / AMC
    time_of_day = (earnings.time_of_day or "").upper()
    if "BMO" in time_of_day or "BEFORE" in time_of_day:
        idx_event = idx_ann
    elif "AMC" in time_of_day or "AFTER" in time_of_day or "PM" in time_of_day:
        idx_event = min(len(df) - 1, idx_ann + 1)
    else:
        idx_event = idx_ann

    event_row = df.iloc[idx_event]
    event_date = event_row["date"].to_pydatetime()

    returns: Dict[str, Optional[float]] = {}
    for label, offset in [("T+1", 1), ("T+3", 3), ("T+7", 7)]:
        target_idx = idx_event + offset
        if target_idx >= len(df):
            returns[label] = None
            continue
        price = float(df.iloc[target_idx].get("close"))
        ret = safe_div(price, baseline_price)
        if ret is None:
            returns[label] = None
        else:
            returns[label] = (ret - 1.0) * 100.0

    return EventStudyResult(
        baseline_date=baseline_date,
        baseline_price=baseline_price,
        event_date=event_date,
        returns=returns,
    )


# ---------------------------
# Transcript summarisation
# ---------------------------


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "have",
    "from",
    "will",
    "our",
    "are",
    "but",
    "not",
    "into",
    "your",
    "about",
    "also",
    "they",
    "them",
    "there",
    "their",
    "been",
    "what",
    "when",
    "where",
    "which",
    "while",
    "over",
    "year",
    "years",
    "quarter",
    "quarters",
}


def _normalize_lang(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    mapping = {
        "en": "en",
        "en-US": "en",
        "en-GB": "en",
        "zh-TW": "zh-TW",
        "zh-Hant": "zh-TW",
        "zh": "zh-TW",
        "ja": "ja",
        "jp": "ja",
        "de": "de",
        "de-DE": "de",
    }
    return mapping.get(lang, "en")


def summarise_transcript_with_openai(text: str, lang: str = "en") -> Optional[str]:
    lang = _normalize_lang(lang)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        # Truncate very long transcripts to keep payload reasonable
        max_chars = 16_000
        if len(text) > max_chars:
            text = text[:max_chars]
        if lang == "zh-TW":
            summary_instr = (
                "Summarize the following US earnings call transcript into around 10 bullet points "
                "in Traditional Chinese, covering demand, pricing, inventory, costs, guidance and risks."
            )
        elif lang == "ja":
            summary_instr = (
                "Summarize the following US earnings call transcript into around 10 bullet points "
                "in Japanese, covering demand, pricing, inventory, costs, guidance and risks."
            )
        elif lang == "de":
            summary_instr = (
                "Summarize the following US earnings call transcript into around 10 bullet points "
                "in German, covering demand, pricing, inventory, costs, guidance and risks."
            )
        else:
            summary_instr = (
                "Summarize the following US earnings call transcript into around 10 bullet points "
                "in English, covering demand, pricing, inventory, costs, guidance and risks."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior equity analyst. " + summary_instr,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            logging.warning("OpenAI API error %s: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        content = choices[0]["message"]["content"]
        return content.strip()
    except Exception as exc:
        logging.warning("OpenAI summarisation failed: %s", exc)
        return None


def keyword_summary(text: str, top_k: int = 15, lang: str = "en") -> str:
    lang = _normalize_lang(lang)
    import re
    from collections import Counter

    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    counter = Counter(tokens)
    most_common = counter.most_common(top_k)

    # Simple sentence scoring based on keyword hits
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored_sentences: List[Tuple[float, str]] = []
    for sent in sentences:
        s_lower = sent.lower()
        score = sum(freq for word, freq in most_common if word in s_lower)
        if score > 0:
            scored_sentences.append((score, sent.strip()))
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored_sentences[:5]]

    if lang == "zh-TW":
        header = "關鍵詞（fallback 模式）："
        sub_header = "自動句子摘錄："
    elif lang == "ja":
        header = "キーワード（フォールバックモード）："
        sub_header = "自動抽出された文："
    elif lang == "de":
        header = "Schlüsselwörter (Fallback-Modus):"
        sub_header = "Automatisch ausgewählte Sätze:"
    else:
        header = "Keywords (fallback mode):"
        sub_header = "Auto-selected sentences:"

    lines = [header]
    if most_common:
        lines.append(
            ", ".join(
                f"{word} ({freq})"
                for word, freq in most_common
            )
        )
    if top_sentences:
        lines.append("")
        lines.append(sub_header)
        for s in top_sentences:
            lines.append(f"- {s}")
    return "\n".join(lines)


def build_transcript_summary(transcript_df: pd.DataFrame, lang: str = "en") -> str:
    if transcript_df.empty:
        if _normalize_lang(lang) == "zh-TW":
            return "未取得財報逐字稿。"
        if _normalize_lang(lang) == "ja":
            return "決算説明会のトランスクリプトを取得できませんでした。"
        if _normalize_lang(lang) == "de":
            return "Es konnten keine Earnings-Call-Transkripte abgerufen werden."
        return "Earnings call transcript is not available."
    # FMP usually returns a list; content field might be 'content' or 'transcript'
    row = transcript_df.iloc[0]
    text = ""
    for key in ("content", "transcript", "text"):
        if key in row and pd.notna(row.get(key)):
            text = str(row.get(key))
            break
    if not text:
        if _normalize_lang(lang) == "zh-TW":
            return "逐字稿內容為空或不可用。"
        if _normalize_lang(lang) == "ja":
            return "トランスクリプトの内容が空か利用できません。"
        if _normalize_lang(lang) == "de":
            return "Der Inhalt des Transkripts ist leer oder nicht verfügbar."
        return "Transcript content is empty or unavailable."
    llm_summary = summarise_transcript_with_openai(text, lang=lang)
    if llm_summary:
        return llm_summary
    return keyword_summary(text, lang=lang)


# ---------------------------
# Rendering helpers (Rich)
# ---------------------------


def render_basic_info(symbol: str, profile_df: pd.DataFrame) -> None:
    if profile_df.empty:
        console.print(Panel(f"[bold]{symbol}[/bold]\n無法取得公司基本資料。", title="基本資訊"))
        return
    row = profile_df.iloc[0]
    name = row.get("companyName") or row.get("company") or row.get("name") or symbol
    exchange = row.get("exchangeShortName") or row.get("exchange") or "N/A"
    sector = row.get("sector") or "N/A"
    industry = row.get("industry") or "N/A"
    market_cap = row.get("mktCap") or row.get("marketCap")
    fiscal_year_end = row.get("fiscalYear") or row.get("fiscalYearEnd") or row.get("fiscalYearEndDate")

    lines = [
        f"[bold]{name}[/bold] ({symbol})",
        f"Exchange: {exchange}",
        f"Sector / Industry: {sector} / {industry}",
        f"Market Cap: {format_currency_compact(float(market_cap)) if pd.notna(market_cap) else 'N/A'}",
        f"Fiscal Year End: {fiscal_year_end if pd.notna(fiscal_year_end) else 'N/A'}",
    ]
    console.print(Panel("\n".join(lines), title="基本資訊", expand=False))


def render_financial_highlights(
    current_income: Dict[str, Optional[float]],
    yoy_income: Dict[str, Optional[float]],
    qoq_income: Dict[str, Optional[float]],
    current_cf: Dict[str, Optional[float]],
    yoy_cf: Dict[str, Optional[float]],
) -> None:
    table = Table(title="財報重點（單季）", show_lines=True)
    table.add_column("項目")
    table.add_column("本季")
    table.add_column("毛利率 / 營業利益率 / 淨利率", justify="right", no_wrap=True)
    table.add_column("YoY 變化")
    table.add_column("QoQ 變化")

    revenue = current_income["revenue"]

    def row_line(label: str, key: str) -> None:
        cur = current_income[key]
        yoy = yoy_income.get(key)
        qoq = qoq_income.get(key)
        margin = compute_margin(cur, revenue) if key != "revenue" else None
        margin_str = (
            format_pct(margin)
            if margin is not None
            else ("-" if key != "revenue" else "")
        )
        table.add_row(
            label,
            format_currency(cur),
            margin_str,
            format_pct(compute_growth(cur, yoy)),
            format_pct(compute_growth(cur, qoq)),
        )

    row_line("Revenue", "revenue")
    row_line("Gross Profit", "gross_profit")
    row_line("Operating Income", "operating_income")
    row_line("Net Income", "net_income")

    # Cash flow
    table2 = Table(title="現金流量（單季）", show_lines=True)
    table2.add_column("項目")
    table2.add_column("本季")
    table2.add_column("YoY 變化")
    for label, key in [("CFO", "cfo"), ("CAPEX", "capex"), ("FCF", "fcf")]:
        cur = current_cf[key]
        yoy = yoy_cf.get(key)
        table2.add_row(
            label,
            format_currency(cur),
            format_pct(compute_growth(cur, yoy)),
        )

    console.print(table)
    console.print(table2)


def render_balance_trend(trend_df: pd.DataFrame) -> None:
    if trend_df.empty:
        console.print(Panel("無法取得近 4 季資產負債表重點。", title="資產負債表趨勢"))
        return
    df = trend_df.copy()
    df["label"] = df["date"].dt.strftime("%Y-%m-%d")
    cols_map = {
        "cashAndShortTermInvestments": "Cash+ST Inv",
        "cashAndCashEquivalents": "Cash & Equiv",
        "shortTermInvestments": "Short-term Inv",
        "totalDebt": "Total Debt",
        "netDebt": "Net Debt",
        "inventory": "Inventory",
        "netReceivables": "Receivables",
        "totalStockholdersEquity": "Equity",
    }
    table = Table(title="資產負債表重點（近 4 季）", show_lines=True)
    table.add_column("項目")
    for label in df["label"]:
        table.add_column(label, justify="right")

    for col, label in cols_map.items():
        if col not in df.columns:
            continue
        values = [format_currency(v) for v in df[col]]
        table.add_row(label, *values)

    console.print(table)


def render_earnings_highlight(earnings: EarningsHighlight) -> None:
    lines: List[str] = []
    if earnings.announcement_date:
        lines.append(f"公告日期 (UTC): {earnings.announcement_date.strftime('%Y-%m-%d')}")
    if earnings.time_of_day:
        lines.append(f"公告時點: {earnings.time_of_day}")

    lines.append("")
    lines.append("EPS:")
    lines.append(
        f"  Actual: {earnings.eps_actual}, Estimate: {earnings.eps_estimate}, "
        f"Surprise: {earnings.eps_surprise}, Surprise %: {format_pct(earnings.eps_surprise_pct)}"
    )

    lines.append("")
    lines.append("Revenue:")
    lines.append(
        f"  Actual: {format_currency(earnings.revenue_actual)}, "
        f"Estimate: {format_currency(earnings.revenue_estimate)}, "
        f"Surprise %: {format_pct(earnings.revenue_surprise_pct)}"
    )

    console.print(Panel("\n".join(lines), title="EPS / 營收驚喜", expand=False))


def render_event_study(event: EventStudyResult) -> None:
    if event.baseline_date is None or event.baseline_price is None:
        console.print(Panel("無法進行價格事件研究（缺少日價或公告日期）。", title="價格反應"))
        return
    table = Table(title="價格事件研究", show_lines=True)
    table.add_column("項目")
    table.add_column("數值", justify="right")
    table.add_row("Baseline 日 (收盤)", event.baseline_date.strftime("%Y-%m-%d"))
    table.add_row("Baseline 價格", f"{event.baseline_price:.2f}")
    if event.event_date:
        table.add_row("Event 日", event.event_date.strftime("%Y-%m-%d"))
    for label in ["T+1", "T+3", "T+7"]:
        table.add_row(label, format_pct(event.returns.get(label)))
    console.print(table)


def maybe_plot_event_study(
    symbol: str,
    hist_df: pd.DataFrame,
    event: EventStudyResult,
    enable: bool,
    year: int,
    quarter: int,
) -> Optional[Path]:
    if not enable:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        logging.info("matplotlib not available; skip plotting")
        return None
    if event.baseline_date is None:
        return None
    df = hist_df.copy()
    df = df.dropna(subset=["date", "close"])
    if df.empty:
        return None
    # Focus on small window around event
    end = event.baseline_date + timedelta(days=14)
    start = event.baseline_date - timedelta(days=7)
    mask = (df["date"] >= start) & (df["date"] <= end)
    window = df[mask].copy()
    if window.empty:
        return None
    window = window.sort_values("date")
    base_price = float(window.iloc[0]["close"])
    rel_ret = window["close"] / base_price - 1.0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(window["date"], rel_ret * 100.0, marker="o")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{symbol} Event Study Around Earnings ({year}Q{quarter})")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()

    reports_dir = ensure_reports_dir()
    path = reports_dir / f"{symbol.upper()}_{year}Q{quarter}_event.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------
# Markdown report
# ---------------------------


def write_markdown_report(
    symbol: str,
    year: int,
    quarter: int,
    profile_df: pd.DataFrame,
    income_current: Dict[str, Optional[float]],
    income_yoy: Dict[str, Optional[float]],
    income_qoq: Dict[str, Optional[float]],
    cf_current: Dict[str, Optional[float]],
    cf_yoy: Dict[str, Optional[float]],
    earnings: EarningsHighlight,
    event: EventStudyResult,
    transcript_summary: str,
) -> Path:
    reports_dir = ensure_reports_dir()
    path = reports_dir / f"{symbol.upper()}_{year}Q{quarter}.md"
    name = symbol
    if not profile_df.empty:
        row = profile_df.iloc[0]
        name = row.get("companyName") or row.get("company") or row.get("name") or symbol

    lines: List[str] = []
    lines.append(f"# {name} ({symbol.upper()}) - {year}Q{quarter} Earnings Analysis")
    lines.append("")
    lines.append(f"_Generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_")
    lines.append("")
    lines.append("## Basic Info")
    if not profile_df.empty:
        row = profile_df.iloc[0]
        exchange = row.get("exchangeShortName") or row.get("exchange") or "N/A"
        sector = row.get("sector") or "N/A"
        industry = row.get("industry") or "N/A"
        market_cap = row.get("mktCap") or row.get("marketCap")
        fiscal_year_end = row.get("fiscalYear") or row.get("fiscalYearEnd") or row.get("fiscalYearEndDate")
        lines.append(f"- Exchange: {exchange}")
        lines.append(f"- Sector / Industry: {sector} / {industry}")
        lines.append(f"- Market Cap: {format_currency_compact(float(market_cap)) if pd.notna(market_cap) else 'N/A'}")
        lines.append(f"- Fiscal Year End: {fiscal_year_end if pd.notna(fiscal_year_end) else 'N/A'}")
    else:
        lines.append("- Company profile not available.")

    lines.append("")
    lines.append("## Income Statement & Cash Flow (Quarterly)")
    lines.append("")

    def metric_line(label: str, key: str) -> None:
        cur = income_current[key]
        yoy = income_yoy.get(key)
        qoq = income_qoq.get(key)
        lines.append(
            f"- **{label}**: {format_currency(cur)} "
            f"(YoY {format_pct(compute_growth(cur, yoy))}, "
            f"QoQ {format_pct(compute_growth(cur, qoq))})"
        )

    metric_line("Revenue", "revenue")
    metric_line("Gross Profit", "gross_profit")
    metric_line("Operating Income", "operating_income")
    metric_line("Net Income", "net_income")

    lines.append("")
    lines.append(
        f"- **CFO**: {format_currency(cf_current['cfo'])} "
        f"(YoY {format_pct(compute_growth(cf_current['cfo'], cf_yoy.get('cfo')))})"
    )
    lines.append(
        f"- **CAPEX**: {format_currency(cf_current['capex'])} "
        f"(YoY {format_pct(compute_growth(cf_current['capex'], cf_yoy.get('capex')))})"
    )
    lines.append(
        f"- **FCF**: {format_currency(cf_current['fcf'])} "
        f"(YoY {format_pct(compute_growth(cf_current['fcf'], cf_yoy.get('fcf')))})"
    )

    lines.append("")
    lines.append("## EPS & Revenue Surprise")
    if earnings.announcement_date:
        lines.append(f"- Announcement Date (UTC): {earnings.announcement_date.strftime('%Y-%m-%d')}")
    if earnings.time_of_day:
        lines.append(f"- Time of Day: {earnings.time_of_day}")
    lines.append(
        f"- EPS: actual {earnings.eps_actual}, estimate {earnings.eps_estimate}, "
        f"surprise {earnings.eps_surprise}, surprise % {format_pct(earnings.eps_surprise_pct)}"
    )
    lines.append(
        f"- Revenue: actual {format_currency(earnings.revenue_actual)}, "
        f"estimate {format_currency(earnings.revenue_estimate)}, "
        f"surprise % {format_pct(earnings.revenue_surprise_pct)}"
    )

    lines.append("")
    lines.append("## Event Study (Price Reaction)")
    if event.baseline_date and event.baseline_price:
        lines.append(
            f"- Baseline close ({event.baseline_date.strftime('%Y-%m-%d')}): {event.baseline_price:.2f}"
        )
        if event.event_date:
            lines.append(f"- Event date: {event.event_date.strftime('%Y-%m-%d')}")
        for label in ["T+1", "T+3", "T+7"]:
            lines.append(f"- {label} cumulative return: {format_pct(event.returns.get(label))}")
    else:
        lines.append("- Not available (missing prices or earnings date).")

    lines.append("")
    lines.append("## Transcript Highlights")
    lines.append("")
    lines.append(textwrap.dedent(transcript_summary).strip())

    lines.append("")
    lines.append("## Data Source")
    lines.append("- Financial data: https://financialmodelingprep.com/stable")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------
# CLI Flow
# ---------------------------


def pick_symbol_interactively(client: FMPClient) -> str:
    query = Prompt.ask("請輸入公司名稱或代號關鍵字")
    console.print("搜尋中，請稍候…")
    df = client.search_name(query)
    if df.empty:
        console.print("[red]找不到符合的標的，請嘗試其他關鍵字。[/red]")
        raise typer.Exit(code=1)
    df = df.head(10)
    table = Table(title="搜尋結果（前 10 筆）", show_lines=True)
    table.add_column("#", justify="right")
    for col in ("symbol", "name", "exchangeShortName"):
        if col not in df.columns:
            df[col] = ""
    table.add_column("Symbol")
    table.add_column("Name")
    table.add_column("Exchange")
    for idx, row in df.reset_index(drop=True).iterrows():
        table.add_row(str(idx), row["symbol"], row["name"], row["exchangeShortName"])
    console.print(table)
    choice = IntPrompt.ask("請輸入欲分析標的編號", choices=[str(i) for i in range(len(df))])
    symbol = str(df.iloc[int(choice)]["symbol"]).upper()
    return symbol


def pick_quarter_interactively(client: FMPClient, symbol: str) -> QuarterSelection:
    console.print(f"讀取 {symbol} 財報逐字稿可用季度…")
    df = client.transcript_dates(symbol)
    if df.empty:
        console.print(
            "[yellow]找不到逐字稿日期清單，將改以 earnings 清單推估季度，但可能無法提供逐字稿摘要。[/yellow]"
        )
        earnings_df = client.earnings(symbol)
        if earnings_df.empty:
            console.print("[red]連 earnings 清單也取得失敗，無法選擇季度。[/red]")
            raise typer.Exit(code=1)
        # derive year/quarter from fiscalDateEnding or date
        options: List[Tuple[int, int]] = []
        for _, row in earnings_df.iterrows():
            if "fiscalDateEnding" in row and pd.notna(row["fiscalDateEnding"]):
                y, q = infer_quarter_from_date(str(row["fiscalDateEnding"]))
            else:
                date_val = row.get("date")
                if isinstance(date_val, pd.Timestamp):
                    y, q = date_val.year, (date_val.month - 1) // 3 + 1
                else:
                    y, q = infer_quarter_from_date(str(date_val))
            if y is None or q is None:
                continue
            options.append((y, q))
        options = sorted(set(options), reverse=True)
        table = Table(title="可用季度（依 earnings 推估）", show_lines=True)
        table.add_column("#", justify="right")
        table.add_column("Year")
        table.add_column("Quarter")
        for idx, (y, q) in enumerate(options):
            table.add_row(str(idx), str(y), f"Q{q}")
        console.print(table)
        choice = IntPrompt.ask("請選擇季度編號", choices=[str(i) for i in range(len(options))])
        y, q = options[int(choice)]
        return QuarterSelection(symbol=symbol, year=y, quarter=q)

    # Use transcript dates list
    if "date" in df.columns:
        df = df.sort_values("date", ascending=False)
    table = Table(title="可用季度（逐字稿）", show_lines=True)
    table.add_column("#", justify="right")
    if "year" not in df.columns:
        df["year"] = df["fiscalYear"] if "fiscalYear" in df.columns else df["date"].dt.year
    if "quarter" not in df.columns:
        df["quarter"] = df["fiscalQuarter"] if "fiscalQuarter" in df.columns else (
            ((df["date"].dt.month - 1) // 3 + 1)
        )
    table.add_column("Year")
    table.add_column("Quarter")
    table.add_column("Date")

    df = df.head(12).reset_index(drop=True)
    for idx, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if isinstance(row.get("date"), pd.Timestamp) else str(
            row.get("date")
        )
        table.add_row(str(idx), str(row["year"]), f"Q{int(row['quarter'])}", date_str)
    console.print(table)

    if Confirm.ask("是否以最新一季為主？", default=True):
        row = df.iloc[0]
        return QuarterSelection(symbol=symbol, year=int(row["year"]), quarter=int(row["quarter"]))

    choice = IntPrompt.ask("請選擇季度編號", choices=[str(i) for i in range(len(df))])
    row = df.iloc[int(choice)]
    return QuarterSelection(symbol=symbol, year=int(row["year"]), quarter=int(row["quarter"]))


def ensure_quarter_selection(
    client: FMPClient,
    symbol: str,
    year: Optional[int],
    quarter: Optional[int],
    latest: bool,
) -> QuarterSelection:
    if year is not None and quarter is not None:
        return QuarterSelection(symbol=symbol.upper(), year=year, quarter=quarter)
    if latest:
        # Try use transcript dates; if none, fallback to earnings
        df = client.transcript_dates(symbol)
        if not df.empty:
            if "date" in df.columns:
                df = df.sort_values("date", ascending=False)
            if "year" not in df.columns:
                df["year"] = df["fiscalYear"] if "fiscalYear" in df.columns else df["date"].dt.year
            if "quarter" not in df.columns:
                df["quarter"] = df["fiscalQuarter"] if "fiscalQuarter" in df.columns else (
                    ((df["date"].dt.month - 1) // 3 + 1)
                )
            row = df.iloc[0]
            return QuarterSelection(symbol=symbol.upper(), year=int(row["year"]), quarter=int(row["quarter"]))
        earnings_df = client.earnings(symbol)
        if earnings_df.empty:
            console.print("[red]無法從逐字稿或 earnings 推出最新季度，請改用互動式選擇。[/red]")
            return pick_quarter_interactively(client, symbol)
        sorted_df = earnings_df.sort_values("date", ascending=False)
        row = sorted_df.iloc[0]
        fy = int(row.get("fy"))
        fq = int(row.get("fq"))
        return QuarterSelection(symbol=symbol.upper(), year=fy, quarter=fq)
    return pick_quarter_interactively(client, symbol)


def run_analysis(
    client: FMPClient,
    selection: QuarterSelection,
    save_report: bool,
    enable_plot: bool,
) -> None:
    # CLI 預設使用繁體中文逐字稿摘要
    analysis = build_analysis(client, selection, lang="zh-TW")
    symbol = analysis["symbol"]
    year = analysis["year"]
    quarter = analysis["quarter"]

    console.rule(f"[bold green]{symbol} {year} Q{quarter} 分析[/bold green]")

    profile_df = analysis["profile_df"]
    income_current = analysis["income_current"]
    income_yoy = analysis["income_yoy"]
    income_qoq = analysis["income_qoq"]
    cf_current = analysis["cf_current"]
    cf_yoy = analysis["cf_yoy"]
    balance_df = analysis["balance_df"]
    earnings_highlight = analysis["earnings"]
    event_result = analysis["event"]
    transcript_summary = analysis["transcript_summary"]
    hist_df = analysis["hist_df"]

    render_basic_info(symbol, profile_df)
    render_financial_highlights(income_current, income_yoy, income_qoq, cf_current, cf_yoy)
    render_balance_trend(extract_balance_trend(balance_df))
    render_earnings_highlight(earnings_highlight)
    render_event_study(event_result)
    console.print(Panel(transcript_summary, title="逐字稿重點", expand=False))

    plot_path = maybe_plot_event_study(symbol, hist_df, event_result, enable_plot, year, quarter)
    if plot_path:
        console.print(f"[green]已產生事件窗折線圖：{plot_path}[/green]")

    if save_report:
        report_path = write_markdown_report(
            symbol=symbol,
            year=year,
            quarter=quarter,
            profile_df=profile_df,
            income_current=income_current,
            income_yoy=income_yoy,
            income_qoq=income_qoq,
            cf_current=cf_current,
            cf_yoy=cf_yoy,
            earnings=earnings_highlight,
            event=event_result,
            transcript_summary=transcript_summary,
        )
        console.print(f"[green]已輸出報告：{report_path}[/green]")


@app.command()
def main(
    symbol: Optional[str] = typer.Option(
        None,
        "--symbol",
        "-s",
        help="股票代號（如：AAPL）；若未提供則以關鍵字搜尋互動選擇。",
    ),
    year: Optional[int] = typer.Option(
        None,
        "--year",
        "-y",
        help="財報年份（如：2024）。",
    ),
    quarter: Optional[int] = typer.Option(
        None,
        "--quarter",
        "-q",
        min=1,
        max=4,
        help="財報季度（1-4）。",
    ),
    latest: bool = typer.Option(
        True,
        "--latest/--no-latest",
        help="是否以最新一季為主（若未指定 year/quarter）。",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="是否輸出 Markdown 報告。",
    ),
    plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="是否產生事件窗折線圖（若無圖形環境或未安裝 matplotlib 則自動略過）。",
    ),
) -> None:
    """
    互動式美股財報分析 CLI。
    """
    configure_logging()

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        console.print(
            "[red]未設定 FMP_API_KEY，請先設定環境變數，例如：[/red]\n"
            "  export FMP_API_KEY=\"YOUR_FMP_KEY\""
        )
        raise typer.Exit(code=1)

    client = FMPClient(api_key=api_key)

    if symbol is None:
        symbol = pick_symbol_interactively(client)
    else:
        symbol = symbol.upper()

    selection = ensure_quarter_selection(client, symbol, year, quarter, latest)
    run_analysis(client, selection, save_report=save, enable_plot=plot)


if __name__ == "__main__":
    try:
        app()
    except FMPError as exc:
        console.print(f"[red]FMP API 錯誤：{exc}[/red]")
        sys.exit(1)
