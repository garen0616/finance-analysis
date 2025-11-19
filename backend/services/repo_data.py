import csv
import glob
import os
import re
from typing import Dict, List, Optional
from dateutil import parser

_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "EarningsCallAgenticRag"))
DATA_ROOT = os.environ.get("DATA_ROOT", _DEFAULT_ROOT)

TRANSCRIPT_FILES = ["maec_transcripts.csv", "merged_data_nasdaq.csv", "merged_data_nyse.csv"]
TRANSCRIPT_COLS = ["transcript", "text", "content", "call_text"]
DATE_COLS = ["date", "earnings_date", "announce_date", "call_date"]
TICKER_COLS = ["symbol", "ticker"]
YEAR_COLS = ["fiscal_year", "fiscalYear", "year"]
QTR_COLS = ["fiscal_quarter", "fiscalQuarter", "quarter", "fiscalPeriod", "q"]

INCOME_KEYS = {
    "revenue": ["Main business income", "Operating Income", "Operating revenue", "Revenue", "Total operating income"],
    "grossProfit": ["Gross profit"],
    "operatingIncome": ["Operating Profit"],
    "netIncome": [
        "net profit attributable to common shareholders",
        "net profit attributable to parent company shareholders",
        "Net profit",
    ],
    "epsdiluted": [
        "Diluted earnings per share-Common stock",
        "basic earnings per share-Common stock",
        "Diluted earnings per share",
    ],
}

BALANCE_KEYS = {
    "totalAssets": ["Total assets", "total assets"],
    "totalLiabilities": ["Total liabilities", "total liabilities"],
    "totalShareholderEquity": ["Total shareholders' equity", "total shareholders' equity"],
}

CASHFLOW_KEYS = {
    "netCashProvidedByOperatingActivities": ["Net Cash Flow from Operating Activities"],
    "capitalExpenditure": ["Purchase of Fixed Assets", "Purchase of property, plant and equipment"],
}


def _lower_map(headers: List[str]) -> Dict[str, str]:
    return {h.lower(): h for h in headers}


class RepoDataLoader:
    def __init__(self, root: str = DATA_ROOT):
        self.root = os.path.abspath(root)

    def _datasets(self) -> List[str]:
        ds = []
        for fn in TRANSCRIPT_FILES:
            path = os.path.join(self.root, fn)
            if os.path.isfile(path):
                ds.append(path)
        return ds

    def list_datasets(self) -> List[Dict]:
        return [{"name": os.path.basename(p), "path": p, "size": os.path.getsize(p)} for p in self._datasets()]

    def _get_dataset_path(self, dataset: str) -> str:
        if os.path.isabs(dataset):
            return dataset
        for path in self._datasets():
            if os.path.basename(path) == dataset:
                return path
        raise FileNotFoundError(f"dataset not found: {dataset}")

    def _read_rows(self, path: str) -> List[Dict]:
        rows: List[Dict] = []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append({k: v for k, v in r.items()})
        except FileNotFoundError:
            return []
        return rows

    def _read_table(self, path: str) -> List[List[str]]:
        table: List[List[str]] = []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        table.append([cell.strip() for cell in row])
        except FileNotFoundError:
            return []
        return table

    def _table_to_records(self, table: List[List[str]]) -> List[Dict]:
        if not table or len(table[0]) <= 1:
            return []
        headers = [h.strip() for h in table[0][1:]]
        records: List[Dict] = []
        for idx, head in enumerate(headers):
            rec: Dict[str, str] = {"period": head}
            for row in table[1:]:
                label = row[0].strip()
                if not label:
                    continue
                value = row[idx + 1] if idx + 1 < len(row) else ""
                rec[label] = value
            records.append(rec)
        return records

    def _find_col(self, headers: List[str], candidates: List[str]) -> Optional[str]:
        lower_map = _lower_map(headers)
        for c in candidates:
            if c in headers:
                return c
            lc = c.lower()
            if lc in lower_map:
                return lower_map[lc]
        return None

    def _parse_int(self, value) -> Optional[int]:
        if value is None:
            return None
        try:
            s = str(value).strip()
            if not s:
                return None
            s = s.replace("Q", "").replace("q", "")
            if s.upper().startswith("FY"):
                s = s[2:]
            return int(float(s))
        except Exception:
            return None

    def _parse_date(self, value) -> Optional[str]:
        if value is None:
            return None
        try:
            return parser.parse(str(value)).date().isoformat()
        except Exception:
            return None

    def _parse_period(self, value):
        if not value:
            return (None, None)
        s = str(value).strip()
        match = re.search(r"Q\s*([1-4])", s, re.IGNORECASE)
        quarter = int(match.group(1)) if match else None
        year_match = re.search(r"(20\d{2}|19\d{2})", s)
        year = int(year_match.group(1)) if year_match else None
        return (year, quarter)

    def _infer_period_from_date(self, value):
        try:
            dt = parser.parse(value)
            return dt.year, ((dt.month - 1) // 3) + 1
        except Exception:
            return (None, None)

    def _parse_number(self, value):
        if value is None:
            return None
        s = str(value).strip()
        if not s or s in {"--", "-", "NA"}:
            return None
        multiplier = 1.0
        tokens = [
            ("Hundred million", 1e8),
            ("Ten thousand", 1e4),
            ("billion", 1e9),
            ("million", 1e6),
            ("thousand", 1e3),
        ]
        for token, mult in tokens:
            if token in s:
                multiplier = mult
                s = s.replace(token, "")
                break
        s = s.replace(",", "")
        if s.endswith("%"):
            s = s[:-1]
        try:
            return float(s) * multiplier
        except ValueError:
            return None

    def _extract_metric(self, rec: Dict[str, str], names: List[str]):
        keys = {k.lower(): k for k in rec.keys() if isinstance(k, str)}
        for name in names:
            low = name.lower()
            if low in keys:
                return self._parse_number(rec.get(keys[low]))
        return None

    def _statement_paths(self, symbol: str):
        fin_dir = os.path.join(self.root, "financial_statements")
        if not os.path.isdir(fin_dir):
            return (None, None, None)

        def find_file(kind_options: List[str]):
            for kind in kind_options:
                pat = os.path.join(fin_dir, f"{symbol.upper()}_{kind}*.csv")
                matches = glob.glob(pat)
                if matches:
                    return matches[0]
            return None

        income = find_file(["income", "income_statement"])
        balance = find_file(["balance", "balance_sheet"])
        cashflow = find_file(["cashflow", "cash_flow", "cash_flow_statement"])
        return income, balance, cashflow

    def _normalize_records(self, records: List[Dict], kind: str):
        norm = []
        for rec in records:
            out = dict(rec)
            period = rec.get("period")
            year, quarter = self._infer_period_from_date(period) if period else (None, None)
            if not year or not quarter:
                py, pq = self._parse_period(rec.get("Financial Report Type"))
                year = year or py
                quarter = quarter or pq
            out["fiscalDateEnding"] = period
            out["calendarYear"] = year
            out["fiscalYear"] = year
            out["fiscalQuarter"] = quarter
            out["period"] = rec.get("Financial Report Type") or (f"Q{quarter}" if quarter else None)
            out["fiscalPeriod"] = out["period"]
            if kind == "income":
                out["revenue"] = self._extract_metric(rec, INCOME_KEYS["revenue"])
                out["grossProfit"] = self._extract_metric(rec, INCOME_KEYS["grossProfit"])
                out["operatingIncome"] = self._extract_metric(rec, INCOME_KEYS["operatingIncome"])
                out["netIncome"] = self._extract_metric(rec, INCOME_KEYS["netIncome"])
                out["epsdiluted"] = self._extract_metric(rec, INCOME_KEYS["epsdiluted"])
            elif kind == "balance":
                out["totalAssets"] = self._extract_metric(rec, BALANCE_KEYS["totalAssets"])
                out["totalLiabilities"] = self._extract_metric(rec, BALANCE_KEYS["totalLiabilities"])
                out["totalShareholderEquity"] = self._extract_metric(rec, BALANCE_KEYS["totalShareholderEquity"])
            elif kind == "cashflow":
                out["netCashProvidedByOperatingActivities"] = self._extract_metric(rec, CASHFLOW_KEYS["netCashProvidedByOperatingActivities"])
                out["capitalExpenditure"] = self._extract_metric(rec, CASHFLOW_KEYS["capitalExpenditure"])
            norm.append(out)
        norm.sort(key=lambda x: x.get("fiscalDateEnding") or "", reverse=True)
        return norm

    def _filter_records(self, recs: List[Dict], year: Optional[int], quarter: Optional[int]):
        if not recs:
            return []
        filtered = recs
        if year:
            filtered = [r for r in filtered if r.get("fiscalYear") == year]
        if quarter:
            filtered = [r for r in filtered if r.get("fiscalQuarter") == quarter]
        if not filtered:
            return recs[:12]
        return filtered[:12]

    def list_tickers(self, dataset: str) -> List[Dict]:
        path = self._get_dataset_path(dataset)
        rows = self._read_rows(path)
        if not rows:
            return []
        headers = list(rows[0].keys())
        tcol = self._find_col(headers, TICKER_COLS)
        if not tcol:
            return []
        counts: Dict[str, int] = {}
        for r in rows:
            sym = (r.get(tcol) or "").upper()
            if not sym:
                continue
            counts[sym] = counts.get(sym, 0) + 1
        filtered = []
        for k, v in counts.items():
            income_p, bal_p, cf_p = self._statement_paths(k)
            if not (income_p or bal_p or cf_p):
                continue
            filtered.append({"symbol": k, "count": v})
        return filtered

    def list_periods(self, dataset: str, symbol: str) -> List[Dict]:
        path = self._get_dataset_path(dataset)
        rows = self._read_rows(path)
        if not rows:
            return []
        headers = list(rows[0].keys())
        tcol = self._find_col(headers, TICKER_COLS)
        ycol = self._find_col(headers, YEAR_COLS)
        qcol = self._find_col(headers, QTR_COLS)
        dcol = self._find_col(headers, DATE_COLS)
        out = []
        for r in rows:
            sym = (r.get(tcol) or "").upper() if tcol else ""
            if sym != symbol.upper():
                continue
            fy = self._parse_int(r.get(ycol)) if ycol else None
            fq = self._parse_int(r.get(qcol)) if qcol else None
            if (fy is None or fq is None) and qcol:
                py, pq = self._parse_period(r.get(qcol))
                fy = fy or py
                fq = fq or pq
            dt = self._parse_date(r.get(dcol)) if dcol else None
            out.append({"fiscalYear": fy, "fiscalQuarter": fq, "periodEnd": dt})
        return out

    def load_transcript(self, dataset: str, symbol: str, year: Optional[int], quarter: Optional[int]):
        path = self._get_dataset_path(dataset)
        rows = self._read_rows(path)
        if not rows:
            return None
        headers = list(rows[0].keys())
        tcol = self._find_col(headers, TICKER_COLS)
        ycol = self._find_col(headers, YEAR_COLS)
        qcol = self._find_col(headers, QTR_COLS)
        tfield = self._find_col(headers, TRANSCRIPT_COLS)
        dfield = self._find_col(headers, DATE_COLS)
        for r in rows:
            sym = (r.get(tcol) or "").upper() if tcol else ""
            if sym != symbol.upper():
                continue
            fy = self._parse_int(r.get(ycol)) if ycol else None
            fq = self._parse_int(r.get(qcol)) if qcol else None
            if (fy is None or fq is None) and qcol:
                py, pq = self._parse_period(r.get(qcol))
                fy = fy or py
                fq = fq or pq
            if year and fy and fy != year:
                continue
            if quarter and fq and fq != quarter:
                continue
            text = (r.get(tfield) or "") if tfield else ""
            dt = self._parse_date(r.get(dfield)) if dfield else None
            return {"text": text, "date": dt}
        return None

    def load_statements(self, symbol: str, year: Optional[int], quarter: Optional[int]):
        fin_dir = os.path.join(self.root, "financial_statements")
        if not os.path.isdir(fin_dir):
            return None

        income_p, bal_p, cf_p = self._statement_paths(symbol)
        if not income_p and not bal_p and not cf_p:
            return None

        def process(path: Optional[str], kind: str):
            if not path:
                return []
            return self._normalize_records(self._table_to_records(self._read_table(path)), kind)

        income = process(income_p, "income")
        balance = process(bal_p, "balance")
        cashflow = process(cf_p, "cashflow")
        if not income and not balance and not cashflow:
            return None
        return {
            "income_df": self._filter_records(income, year, quarter),
            "balance_df": self._filter_records(balance, year, quarter),
            "cashflow_df": self._filter_records(cashflow, year, quarter),
        }
