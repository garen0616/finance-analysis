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
