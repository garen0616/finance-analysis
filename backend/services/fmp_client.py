import os
from datetime import datetime, timedelta
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class FMPClient:
    def __init__(self, api_key: str, session: requests.Session | None = None):
        self.api_key = api_key
        self.session = session or requests.Session()
        # Default to stable as requested; allow override via env.
        self.base = os.environ.get("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")

    def _latest_transcript_fallback(self, symbol):
        try:
            dates = self._get(f"{self.base}/earning-call-transcript-dates", {"symbol": symbol})
            if isinstance(dates, list) and dates:
                latest = dates[0]
                year = latest.get("year")
                quarter = latest.get("quarter")
                if year and quarter:
                    return self.get_transcript(symbol, year=year, quarter=quarter, latest=False)
        except Exception:
            return {}
        return {}

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
        # Stable search prefers search-symbol; fall back to legacy search.
        try:
            return self._get(f"{self.base}/search-symbol", {"query": query, "limit": 8})
        except Exception:
            return self._get(f"{self.base}/search", {"query": query, "limit": 8})

    def get_company_profile(self, symbol):
        data = self._get(f"{self.base}/profile", {"symbol": symbol})
        return data[0] if isinstance(data, list) and data else {}

    def get_income_q(self, symbol, limit=12):
        return self._get(f"{self.base}/income-statement", {"symbol": symbol, "period": "quarter", "limit": limit})

    def get_balance_q(self, symbol, limit=12):
        return self._get(f"{self.base}/balance-sheet-statement", {"symbol": symbol, "period": "quarter", "limit": limit})

    def get_cashflow_q(self, symbol, limit=12):
        return self._get(f"{self.base}/cash-flow-statement", {"symbol": symbol, "period": "quarter", "limit": limit})

    def get_earnings_company(self, symbol):
        return self._get(f"{self.base}/earnings", {"symbol": symbol})

    def get_earnings_calendar(self, symbol, from_date, to_date):
        return self._get(f"{self.base}/earnings-calendar", {"from": from_date, "to": to_date, "symbol": symbol})

    def get_hist_prices(self, symbol, start_date, end_date):
        res = self._get(f"{self.base}/historical-price-eod/full", {"symbol": symbol, "from": start_date, "to": end_date})
        if isinstance(res, dict):
            if "historical" in res:
                return res.get("historical", [])
        if isinstance(res, list):
            return res
        return []

    def get_stock_peers(self, symbol):
        try:
            data = self._get(f"{self.base}/stock-peers", {"symbol": symbol})
            if isinstance(data, dict) and "peersList" in data:
                return data.get("peersList") or []
            if isinstance(data, list):
                return data
        except Exception:
            return []
        return []

    def get_transcript(self, symbol, year=None, quarter=None, latest=False):
        if latest:
            try:
                dates = self._get(f"{self.base}/earning-call-transcript-dates", {"symbol": symbol})
                if isinstance(dates, list) and dates:
                    latest_d = dates[0]
                    fy = latest_d.get("fiscalYear") or latest_d.get("year")
                    q = latest_d.get("quarter")
                    if fy and q:
                        return self.get_transcript(symbol, year=fy, quarter=q, latest=False)
            except Exception:
                pass
            try:
                all_t = self._get(f"{self.base}/earning-call-transcript", {"symbol": symbol})
                if isinstance(all_t, list):
                    return all_t[0] if all_t else {}
                return all_t
            except Exception:
                return self._latest_transcript_fallback(symbol)
        if year and quarter:
            return self._get(f"{self.base}/earning-call-transcript", {"symbol": symbol, "year": year, "quarter": quarter})
        return {}
