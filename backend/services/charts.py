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
