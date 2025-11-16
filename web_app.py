import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import (
    FMPClient,
    QuarterSelection,
    build_analysis,
    infer_quarter_from_date,
    format_currency,
    format_currency_compact,
    format_pct,
    compute_growth,
    ensure_reports_dir,
    write_markdown_report,
    maybe_plot_event_study,
)


LANG_CHOICES: Dict[str, str] = {
    "en": "English",
    "zh-TW": "繁體中文",
    "ja": "日本語",
    "de": "Deutsch",
}


def normalize_lang(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    if lang in LANG_CHOICES:
        return lang
    return "en"


app = FastAPI(title="Finance Analysis Web")

templates = Jinja2Templates(directory="templates")
templates.env.globals["format_currency"] = format_currency
templates.env.globals["format_pct"] = format_pct
templates.env.globals["format_currency_compact"] = format_currency_compact
templates.env.globals["compute_growth"] = compute_growth

reports_dir = ensure_reports_dir()
app.mount(
    "/reports",
    StaticFiles(directory=str(reports_dir), check_dir=False),
    name="reports",
)


def get_client() -> FMPClient:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="FMP_API_KEY is not configured")
    return FMPClient(api_key=api_key)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, lang: Optional[str] = "en") -> HTMLResponse:
    lang_code = normalize_lang(lang)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "lang": lang_code,
            "lang_choices": LANG_CHOICES,
        },
    )


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = "", lang: Optional[str] = "en") -> HTMLResponse:
    lang_code = normalize_lang(lang)
    query = q.strip()
    if not query:
        return RedirectResponse(url=f"/?lang={lang_code}", status_code=302)
    client = get_client()
    df = client.search_name(query)
    results: List[Dict[str, str]] = []
    if not df.empty:
        df = df.head(10).reset_index(drop=True)
        for _, row in df.iterrows():
            results.append(
                {
                    "symbol": str(row.get("symbol", "")).upper(),
                    "name": str(row.get("name") or row.get("companyName") or ""),
                    "exchange": str(
                        row.get("exchangeShortName")
                        or row.get("stockExchange")
                        or row.get("exchange")
                        or ""
                    ),
                }
            )

    return templates.TemplateResponse(
        "search_results.html",
        {
            "request": request,
            "query": query,
            "lang": lang_code,
            "results": results,
        },
    )


def _build_quarter_options_from_transcripts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False)
    if "year" not in df.columns:
        if "fiscalYear" in df.columns:
            df["year"] = df["fiscalYear"]
        else:
            df["year"] = df["date"].dt.year
    if "quarter" not in df.columns:
        if "fiscalQuarter" in df.columns:
            df["quarter"] = df["fiscalQuarter"]
        else:
            df["quarter"] = ((df["date"].dt.month - 1) // 3 + 1)
    df = df.dropna(subset=["year", "quarter"])
    df = df.head(12).reset_index(drop=True)

    options: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        date_val = row.get("date")
        if isinstance(date_val, pd.Timestamp):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val) if date_val is not None else ""
        year = int(row["year"])
        quarter = int(row["quarter"])
        options.append(
            {"year": year, "quarter": quarter, "date": date_str},
        )
    # Remove duplicates while preserving order
    seen: set[Tuple[int, int]] = set()
    unique: List[Dict[str, Any]] = []
    for opt in options:
        key = (opt["year"], opt["quarter"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(opt)
    return unique


def _build_quarter_options_from_earnings(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    options: List[Tuple[int, int]] = []
    for _, row in df.iterrows():
        if "fiscalDateEnding" in row and pd.notna(row["fiscalDateEnding"]):
            year, quarter = infer_quarter_from_date(str(row["fiscalDateEnding"]))
        else:
            date_val = row.get("date")
            if isinstance(date_val, pd.Timestamp):
                year = date_val.year
                quarter = (date_val.month - 1) // 3 + 1
            else:
                year, quarter = infer_quarter_from_date(str(date_val))
        if year is None or quarter is None:
            continue
        options.append((year, quarter))
    options = sorted(set(options), reverse=True)
    result: List[Dict[str, Any]] = []
    for year, quarter in options:
        result.append({"year": year, "quarter": quarter, "date": ""})
    return result


@app.get("/quarter", response_class=HTMLResponse)
async def select_quarter(
    request: Request,
    symbol: str,
    lang: Optional[str] = "en",
) -> HTMLResponse:
    lang_code = normalize_lang(lang)
    client = get_client()
    symbol = symbol.upper()
    transcripts_df = client.transcript_dates(symbol)
    options: List[Dict[str, Any]] = []
    source = ""
    if not transcripts_df.empty:
        options = _build_quarter_options_from_transcripts(transcripts_df)
        source = "transcript"
    else:
        earnings_df = client.earnings(symbol)
        if earnings_df.empty:
            raise HTTPException(
                status_code=404,
                detail="No transcript dates or earnings data available for this symbol.",
            )
        options = _build_quarter_options_from_earnings(earnings_df)
        source = "earnings"

    if not options:
        raise HTTPException(status_code=404, detail="No quarters available for this symbol.")

    return templates.TemplateResponse(
        "quarter_select.html",
        {
            "request": request,
            "symbol": symbol,
            "options": options,
            "source": source,
            "lang": lang_code,
        },
    )


@app.get("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    symbol: str,
    yq: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
    save: Optional[str] = "1",
    plot: Optional[str] = "1",
    lang: Optional[str] = "en",
) -> HTMLResponse:
    lang_code = normalize_lang(lang)
    if yq:
        try:
            y_str, q_str = yq.split("-", 1)
            year = int(y_str)
            quarter = int(q_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid yq parameter.")
    if year is None or quarter is None:
        raise HTTPException(status_code=400, detail="year and quarter are required.")
    client = get_client()
    selection = QuarterSelection(symbol=symbol.upper(), year=year, quarter=quarter)
    analysis = build_analysis(client, selection, lang=lang_code)

    report_path = None
    save_flag = save is not None
    plot_flag = plot is not None
    if save_flag:
        report_path = write_markdown_report(
            symbol=analysis["symbol"],
            year=analysis["year"],
            quarter=analysis["quarter"],
            profile_df=analysis["profile_df"],
            income_current=analysis["income_current"],
            income_yoy=analysis["income_yoy"],
            income_qoq=analysis["income_qoq"],
            cf_current=analysis["cf_current"],
            cf_yoy=analysis["cf_yoy"],
            earnings=analysis["earnings"],
            event=analysis["event"],
            transcript_summary=analysis["transcript_summary"],
        )

    plot_url: Optional[str] = None
    if plot_flag:
        img_path = maybe_plot_event_study(
            symbol=analysis["symbol"],
            hist_df=analysis["hist_df"],
            event=analysis["event"],
            enable=True,
            year=analysis["year"],
            quarter=analysis["quarter"],
        )
        if img_path is not None:
            plot_url = f"/reports/{img_path.name}"

    return templates.TemplateResponse(
        "analysis.html",
        {
            "request": request,
            "analysis": analysis,
            "report_path": report_path,
            "plot_url": plot_url,
            "lang": lang_code,
            "lang_choices": LANG_CHOICES,
        },
    )


def markdown_to_pdf(markdown_text: str, output_path: str, title: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    page_width, page_height = A4
    margin = 50
    line_height = 14

    c = canvas.Canvas(output_path, pagesize=A4)
    c.setTitle(title)
    y = page_height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= line_height * 2
    c.setFont("Helvetica", 10)

    for raw_line in markdown_text.splitlines():
        line = raw_line.replace("\t", "    ")
        while line:
            if y < margin:
                c.showPage()
                y = page_height - margin
                c.setFont("Helvetica", 10)
            chunk = line[:110]
            c.drawString(margin, y, chunk)
            y -= line_height
            line = line[110:]
    c.showPage()
    c.save()


@app.get("/download/pdf")
async def download_pdf(
    symbol: str,
    yq: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
    lang: Optional[str] = "en",
) -> FileResponse:
    lang_code = normalize_lang(lang)
    if yq:
        try:
            y_str, q_str = yq.split("-", 1)
            year = int(y_str)
            quarter = int(q_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid yq parameter.")
    if year is None or quarter is None:
        raise HTTPException(status_code=400, detail="year and quarter are required.")

    client = get_client()
    selection = QuarterSelection(symbol=symbol.upper(), year=year, quarter=quarter)
    analysis = build_analysis(client, selection, lang=lang_code)

    md_path = write_markdown_report(
        symbol=analysis["symbol"],
        year=analysis["year"],
        quarter=analysis["quarter"],
        profile_df=analysis["profile_df"],
        income_current=analysis["income_current"],
        income_yoy=analysis["income_yoy"],
        income_qoq=analysis["income_qoq"],
        cf_current=analysis["cf_current"],
        cf_yoy=analysis["cf_yoy"],
        earnings=analysis["earnings"],
        event=analysis["event"],
        transcript_summary=analysis["transcript_summary"],
    )
    text = md_path.read_text(encoding="utf-8")
    pdf_path = md_path.with_suffix(".pdf")
    title = f"{analysis['symbol']} {analysis['year']}Q{analysis['quarter']} Earnings Analysis"
    markdown_to_pdf(text, str(pdf_path), title=title)
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
