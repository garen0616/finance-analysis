from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    source: str = Field("fmp", regex="^(fmp|repo)$")
    dataset: Optional[str] = None
    symbol: str
    mode: str = Field("latest", regex="^(latest|specific)$")
    year: Optional[int] = None
    quarter: Optional[int] = None
    forceRefresh: bool = False
    peers: Optional[List[str]] = None

class AnalyzeResponse(BaseModel):
    analysisId: str
    summary: Dict[str, Any]
    kpis: Dict[str, Any]
    tables: Dict[str, Any]
    charts: Dict[str, str]
    transcriptHighlights: List[str]
    graphEnabled: bool

class BatchRequest(BaseModel):
    symbols: List[str]
    mode: str = Field("latest", regex="^(latest|specific)$")
    year: Optional[int] = None
    quarter: Optional[int] = None

class JobStatus(BaseModel):
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
