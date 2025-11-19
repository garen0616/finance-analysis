import csv
import glob
import os
from typing import Dict, List, Optional
from dateutil import parser

_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "EarningsCallAgenticRag"))
DATA_ROOT = os.environ.get("DATA_ROOT", _DEFAULT_ROOT)

TRANSCRIPT_FILES = ["maec_transcripts.csv", "merged_data_nasdaq.csv", "merged_data_nyse.csv"]
TRANSCRIPT_COLS = ["transcript", "text", "content", "call_text"]
DATE_COLS = ["date", "earnings_date", "announce_date", "call_date"]
TICKER_COLS = ["symbol", "ticker"]
YEAR_COLS = ["fiscal_year", "fiscalYear", "year"]
QTR_COLS = ["fiscal_quarter", "fiscalQuarter", "quarter", "fiscalPeriod"]


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
        return [{"symbol": k, "count": v} for k, v in counts.items()]

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

        def find_file(kind: str):
            pat = os.path.join(fin_dir, f"{symbol.upper()}_{kind}*.csv")
            matches = glob.glob(pat)
            return matches[0] if matches else None

        def load_filtered(path: Optional[str]):
            if not path:
                return []
            rows = self._read_rows(path)
            if not rows:
                return []
            headers = list(rows[0].keys())
            ycol = self._find_col(headers, YEAR_COLS)
            qcol = self._find_col(headers, QTR_COLS)
            filtered = []
            for r in rows:
                fy = self._parse_int(r.get(ycol)) if ycol else None
                fq = self._parse_int(r.get(qcol)) if qcol else None
                if year and fy and fy != year:
                    continue
                if quarter and fq and fq != quarter:
                    continue
                filtered.append(r)
            return filtered

        income_p = find_file("income")
        bal_p = find_file("balance")
        cf_p = find_file("cashflow")
        if not income_p and not bal_p and not cf_p:
            return None
        return {
            "income_df": load_filtered(income_p),
            "balance_df": load_filtered(bal_p),
            "cashflow_df": load_filtered(cf_p),
        }
