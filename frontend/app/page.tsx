"use client";
import { useMemo, useState } from "react";
import useSWR from "swr";
import { apiBase, postJSON, pdfUrl } from "../lib/api";
import Header from "../components/brand/Header";
import StickySummary from "../components/brand/StickySummary";
import ControlBar from "../components/brand/ControlBar";
import KpiCard from "../components/brand/KpiCard";
import ChartCard from "../components/brand/ChartCard";
import DataTable from "../components/brand/DataTable";
import PeersPanel from "../components/brand/PeersPanel";
import GraphPreview from "../components/brand/GraphPreview";
import { Tabs } from "../components/ui/tabs";
import { Skeleton } from "../components/ui/skeleton";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { EmptyState } from "../components/brand/EmptyState";
import { ErrorState } from "../components/brand/ErrorState";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

type Analysis = {
  analysisId: string;
  summary: { title: string; bullets: string[] };
  kpis: any;
  tables: any;
  charts: Record<string, string>;
  transcriptHighlights: string[];
  graphEnabled: boolean;
};

type EventWin = { window: string; returnPct: number; shock: boolean };

export default function Page() {
  const [symbol, setSymbol] = useState("AAPL");
  const [activeTab, setActiveTab] = useState("overview");
  const [analysisId, setAnalysisId] = useState<string | undefined>(undefined);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });

  useSWR(`${apiBase}/api/health`, fetcher);

  const runAnalyze = async (mode: "latest" | "specific", year?: number, quarter?: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await postJSON<Analysis>('/api/analyze', { symbol, mode, year, quarter });
      setAnalysis(res);
      setAnalysisId(res.analysisId);
      if (res.graphEnabled) {
        const g = await fetch(`${apiBase}/api/graph/preview?symbol=${symbol}`).then((r) => r.json());
        setGraphData({ nodes: g.nodes || [], links: g.links || [] });
      } else {
        setGraphData({ nodes: [], links: [] });
      }
    } catch (e: any) {
      setError(e.message || 'failed');
    } finally {
      setLoading(false);
    }
  };

  const kpiItems = useMemo(() => {
    if (!analysis?.kpis) return [];
    const k = analysis.kpis;
    const spark = analysis.tables?.income?.map((r: any) => r.revenue).slice(0, 12).reverse() || [];
    return [
      { label: "Revenue", value: k.revenue, delta: 0, spark },
      { label: "EPS", value: k.epsDiluted, delta: k.surprise?.pct ?? 0, spark: [] },
      { label: "Gross Margin", value: k.grossMarginPct, delta: null, spark: [] },
      { label: "Operating Margin", value: k.operatingMarginPct, delta: null, spark: [] },
      { label: "Net Margin", value: k.netMarginPct, delta: null, spark: [] },
      { label: "FCF", value: k.fcf, delta: null, spark: [] },
    ];
  }, [analysis]);

  const chartImg = (key: string) => analysis?.charts?.[key];

  const eventRows: EventWin[] = analysis?.tables?.eventWindows || [];

  return (
    <div className="animated-grid relative">
      <div className="min-h-screen relative z-10">
        <Header analysisId={analysisId} />
        <StickySummary bullets={analysis?.summary?.bullets || []} />
        <ControlBar symbol={symbol} setSymbol={setSymbol} onGenerate={runAnalyze} />

        <div className="mt-6">
          <Tabs
            active={activeTab}
            onChange={setActiveTab}
            tabs={[
              { key: "overview", label: "Overview" },
              { key: "financials", label: "Financials" },
              { key: "transcript", label: "Transcript" },
              { key: "peers", label: "Peers" },
              { key: "graph", label: "Graph" },
              { key: "data", label: "Data" },
            ]}
          />
        </div>

        {error && <ErrorState onRetry={() => runAnalyze("latest")} />}

        {loading && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 my-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-28" />
            ))}
          </div>
        )}

        {!analysis && !loading && <EmptyState message="Run an analysis to see results." />}

        {analysis && (
          <div className="space-y-6 my-4">
            {activeTab === "overview" && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {kpiItems.map((k) => (
                    <KpiCard key={k.label} {...k} />
                  ))}
                </div>
                <Card className="p-4 space-y-3">
                  <div className="text-sm font-semibold">Event Windows</div>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-slate-500">
                        <th className="py-2">Window</th>
                        <th className="py-2">Return</th>
                        <th className="py-2">Shock</th>
                      </tr>
                    </thead>
                    <tbody>
                      {eventRows.length ? (
                        eventRows.map((e, i) => (
                          <tr key={i} className="border-t border-[var(--line)]">
                            <td className="py-2">{e.window}</td>
                            <td className="py-2">{(e.returnPct ?? 0).toFixed(2)}%</td>
                            <td className="py-2">{e.shock ? "Yes" : "No"}</td>
                          </tr>
                        ))
                      ) : (
                        <tr><td className="py-2 text-slate-500" colSpan={3}>No data</td></tr>
                      )}
                    </tbody>
                  </table>
                </Card>
              </>
            )}

            {activeTab === "financials" && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ChartCard title="Revenue" dataUrl={chartImg("revenueTrendPng")} />
                <ChartCard title="EPS" dataUrl={chartImg("epsTrendPng")} />
                <ChartCard title="Margins" dataUrl={chartImg("marginTrendPng")} />
                <ChartCard title="FCF" dataUrl={chartImg("fcfTrendPng")} />
              </div>
            )}

            {activeTab === "transcript" && (
              <Card className="space-y-2">
                <div className="text-lg font-semibold">Transcript</div>
                <ul className="list-disc pl-5 space-y-1">
                  {analysis.transcriptHighlights?.length ? (
                    analysis.transcriptHighlights.map((h, i) => <li key={i}>{h}</li>)
                  ) : (
                    <li className="text-slate-500">No transcript highlights.</li>
                  )}
                </ul>
              </Card>
            )}

            {activeTab === "peers" && <PeersPanel peers={analysis.tables?.peers || {}} />}

            {activeTab === "graph" && <GraphPreview enabled={analysis.graphEnabled} data={graphData} />}

            {activeTab === "data" && (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <Button onClick={() => analysisId && window.open(pdfUrl(analysisId), "_blank")}>Download PDF</Button>
                  <Button variant="outline">Download CSV</Button>
                </div>
                <DataTable title="Income" rows={analysis.tables?.income || []} />
                <DataTable title="Balance" rows={analysis.tables?.balance || []} />
                <DataTable title="Cashflow" rows={analysis.tables?.cashflow || []} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
