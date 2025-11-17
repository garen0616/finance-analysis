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
