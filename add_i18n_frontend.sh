#!/usr/bin/env bash
set -euo pipefail

# detect frontend dir
if [ -d "repo/frontend" ]; then
  FRONTEND_DIR="repo/frontend"
elif [ -d "frontend" ]; then
  FRONTEND_DIR="frontend"
else
  echo "frontend directory not found" >&2
  exit 1
fi

# check node
node -v >/dev/null 2>&1 || { echo "Node.js >=18 required"; exit 1; }
node - <<'JS'
const v = process.versions.node.split('.').map(Number);
if (v[0] < 18) { console.error("Node.js >=18 required"); process.exit(1); }
JS

cd "$FRONTEND_DIR"

npm install next-themes clsx tailwind-merge

mkdir -p providers hooks i18n components lib

cat > providers/I18nProvider.tsx <<'EOF2'
"use client";
import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import en from "../i18n/en";
import zhTW from "../i18n/zh-TW";

type Locale = "en" | "zh-TW";
type Dict = Record<string, string>;
type Ctx = {
  locale: Locale;
  t: (key: string, vars?: Record<string, string | number>) => string;
  setLocale: (loc: Locale) => void;
  nfmt: (n: number, opts?: Intl.NumberFormatOptions) => string;
  dfmt: (d: string | number | Date, opts?: Intl.DateTimeFormatOptions) => string;
};

const dicts: Record<Locale, Dict> = { en, "zh-TW": zhTW };

const I18nContext = createContext<Ctx | null>(null);

const normalize = (lng: string | null): Locale => {
  if (!lng) return "en";
  const l = lng.toLowerCase();
  if (l.startsWith("zh") || l.includes("hant")) return "zh-TW";
  return "en";
};

function interpolate(str: string, vars?: Record<string, string | number>) {
  if (!vars) return str;
  return Object.keys(vars).reduce(
    (acc, k) => acc.replace(new RegExp(`{{\\s*${k}\\s*}}`, "g"), String(vars[k])),
    str
  );
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    const stored = typeof window !== "undefined" ? localStorage.getItem("locale") : null;
    const initial = normalize(stored || (typeof navigator !== "undefined" ? navigator.language : "en"));
    setLocaleState(initial);
  }, []);

  useEffect(() => {
    if (typeof document !== "undefined") document.documentElement.lang = locale;
    if (typeof localStorage !== "undefined") localStorage.setItem("locale", locale);
  }, [locale]);

  const setLocale = (loc: Locale) => setLocaleState(loc);

  const t = useMemo(
    () => (key: string, vars?: Record<string, string | number>) => {
      const value = dicts[locale][key] ?? dicts["en"][key] ?? key;
      return interpolate(value, vars);
    },
    [locale]
  );

  const nfmt = (n: number, opts?: Intl.NumberFormatOptions) =>
    new Intl.NumberFormat(locale, opts).format(n);
  const dfmt = (d: string | number | Date, opts?: Intl.DateTimeFormatOptions) =>
    new Intl.DateTimeFormat(locale, opts).format(new Date(d));

  const ctx: Ctx = { locale, t, setLocale, nfmt, dfmt };

  return <I18nContext.Provider value={ctx}>{children}</I18nContext.Provider>;
}

export function useI18nCtx() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("I18nProvider missing");
  return ctx;
}
EOF2

cat > hooks/useI18n.ts <<'EOF2'
"use client";
import { useI18nCtx } from "../providers/I18nProvider";
export function useI18n() {
  return useI18nCtx();
}
EOF2

cat > i18n/en.ts <<'EOF2'
const en = {
  "app.title": "Earnings Analysis Dashboard",
  "nav.overview": "Overview",
  "nav.financials": "Financials",
  "nav.transcript": "Transcript",
  "nav.peers": "Peers",
  "nav.graph": "Graph",
  "nav.data": "Data",
  "actions.generate": "Generate",
  "actions.downloadPdf": "Download PDF",
  "actions.refresh": "Refresh",
  "actions.addPeers": "Add peers",
  "controls.search.placeholder": "Search company or symbol…",
  "controls.latest": "Latest",
  "controls.specific": "Specific quarter",
  "controls.year": "Year",
  "controls.quarter": "Quarter",
  "summary.title": "Executive Summary",
  "kpi.revenue": "Revenue",
  "kpi.eps": "EPS",
  "kpi.grossMargin": "Gross Margin",
  "kpi.opMargin": "Operating Margin",
  "kpi.netMargin": "Net Margin",
  "kpi.cfo": "Operating Cash Flow",
  "kpi.capex": "CapEx",
  "kpi.fcf": "Free Cash Flow",
  "kpi.fcfMargin": "FCF Margin",
  "event.title": "Event Windows",
  "event.bmo": "BMO",
  "event.amc": "AMC",
  "event.t1": "T+1",
  "event.t3": "T+3",
  "event.t5": "T+5",
  "event.t20": "T+20",
  "event.shock": "Shock",
  "event.noShock": "No shock",
  "peers.title": "Peer Comparison",
  "peers.median": "Peer median",
  "peers.vsCompany": "Vs company",
  "graph.title": "Graph Preview",
  "data.title": "Raw Data",
  "empty.noData": "No data available",
  "error.network": "Network error, please try again",
  "table.income": "Income Statement",
  "table.balance": "Balance Sheet",
  "table.cashflow": "Cash Flow",
  "table.yoy": "YoY Growth",
  "table.qoq": "QoQ Growth",
  "table.eventWindows": "Event Windows",
  "table.downloadCsv": "Download CSV"
};
export default en;
EOF2

cat > i18n/zh-TW.ts <<'EOF2'
const zhTW = {
  "app.title": "財報洞察儀表板",
  "nav.overview": "總覽",
  "nav.financials": "財務",
  "nav.transcript": "逐字稿",
  "nav.peers": "同業比較",
  "nav.graph": "關聯圖譜",
  "nav.data": "原始資料",
  "actions.generate": "生成分析",
  "actions.downloadPdf": "下載 PDF",
  "actions.refresh": "重新整理",
  "actions.addPeers": "加入同業",
  "controls.search.placeholder": "搜尋公司或代號…",
  "controls.latest": "最新",
  "controls.specific": "指定季度",
  "controls.year": "年度",
  "controls.quarter": "季度",
  "summary.title": "執行摘要",
  "kpi.revenue": "營收",
  "kpi.eps": "每股盈餘（EPS）",
  "kpi.grossMargin": "毛利率",
  "kpi.opMargin": "營益率",
  "kpi.netMargin": "淨利率",
  "kpi.cfo": "營運現金流",
  "kpi.capex": "資本支出",
  "kpi.fcf": "自由現金流",
  "kpi.fcfMargin": "FCF 利潤率",
  "event.title": "事件窗表現",
  "event.bmo": "盤前（BMO）",
  "event.amc": "盤後（AMC）",
  "event.t1": "T+1",
  "event.t3": "T+3",
  "event.t5": "T+5",
  "event.t20": "T+20",
  "event.shock": "波動達閾值",
  "event.noShock": "未達閾值",
  "peers.title": "同業比較",
  "peers.median": "同業中位數",
  "peers.vsCompany": "相對本公司",
  "graph.title": "關聯圖譜預覽",
  "data.title": "原始資料",
  "empty.noData": "目前沒有資料",
  "error.network": "連線異常，請稍後再試",
  "table.income": "損益表",
  "table.balance": "資產負債表",
  "table.cashflow": "現金流量表",
  "table.yoy": "年增率（YoY）",
  "table.qoq": "季增率（QoQ）",
  "table.eventWindows": "事件窗",
  "table.downloadCsv": "下載 CSV"
};
export default zhTW;
EOF2

cat > components/LanguageToggle.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";
import clsx from "clsx";

export default function LanguageToggle() {
  const { locale, setLocale, t } = useI18n();
  const opts: { label: string; value: "en" | "zh-TW" }[] = [
    { label: "EN", value: "en" },
    { label: "繁", value: "zh-TW" },
  ];
  return (
    <div className="inline-flex border rounded-full overflow-hidden">
      {opts.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={locale === o.value}
          aria-label={t("app.title")}
          onClick={() => setLocale(o.value)}
          className={clsx(
            "px-3 py-1 text-sm",
            locale === o.value ? "bg-indigo-600 text-white" : "bg-transparent text-gray-700 dark:text-gray-200"
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
EOF2

cat > components/BrandHeader.tsx <<'EOF2'
"use client";
import LanguageToggle from "./LanguageToggle";
import { useI18n } from "../hooks/useI18n";

export default function BrandHeader() {
  const { t } = useI18n();
  return (
    <header className="flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-semibold">{t("app.title")}</h1>
        <p className="text-sm text-gray-500">FMP + RAG + PDF</p>
      </div>
      <div className="flex items-center gap-3">
        <LanguageToggle />
      </div>
    </header>
  );
}
EOF2

cat > lib/format.ts <<'EOF2'
import { useI18n } from "../hooks/useI18n";

export function useFormat() {
  const { locale } = useI18n();
  const formatNumber = (n: number, style: "decimal" | "currency" | "percent" = "decimal", currency = "USD") =>
    new Intl.NumberFormat(locale, { style, currency, maximumFractionDigits: style === "percent" ? 2 : 2 }).format(n);
  const formatSignedPercent = (x: number) => {
    const sign = x > 0 ? "+" : "";
    return `${sign}${formatNumber(x / 100, "percent")}`;
  };
  const formatDate = (iso: string) => new Intl.DateFormat(locale, { year: "numeric", month: "2-digit", day: "2-digit" } as any).format(new Date(iso));
  return { formatNumber, formatSignedPercent, formatDate };
}
EOF2

cat > app/layout.tsx <<'EOF2'
import "./globals.css";
import type { ReactNode } from "react";
import { I18nProvider } from "../providers/I18nProvider";

export const metadata = {
  title: "Earnings Analysis",
  description: "Earnings analytics with FMP + RAG",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen">
        <I18nProvider>
          {children}
        </I18nProvider>
      </body>
    </html>
  );
}
EOF2

cat > app/page.tsx <<'EOF2'
"use client";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { useI18n } from "../hooks/useI18n";
import BrandHeader from "../components/BrandHeader";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

type Analysis = {
  analysisId: string;
  summary: { title: string; bullets: string[] };
  kpis: any;
  tables: any;
  charts: Record<string, string>;
  transcriptHighlights: string[];
  graphEnabled: boolean;
};

const backend = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function Page() {
  const { t } = useI18n();
  const [symbol, setSymbol] = useState("AAPL");
  const [mode, setMode] = useState<"latest" | "specific">("latest");
  const [year, setYear] = useState<number | undefined>(undefined);
  const [quarter, setQuarter] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [symbols, setSymbols] = useState<any[]>([]);
  const [query, setQuery] = useState("");
  const [graphData, setGraphData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });

  useEffect(() => {
    if (query.length < 1) return;
    const tmr = setTimeout(async () => {
      const res = await fetch(`${backend}/api/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      setSymbols(data || []);
    }, 250);
    return () => clearTimeout(tmr);
  }, [query]);

  const generate = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${backend}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, mode, year, quarter }),
      });
      const data = await res.json();
      setAnalysis(data);
      if (data.graphEnabled) {
        const g = await fetch(`${backend}/api/graph/preview?symbol=${symbol}`);
        const gd = await g.json();
        setGraphData({ nodes: gd.nodes || [], links: gd.links || [] });
      }
    } finally {
      setLoading(false);
    }
  };

  const downloadPdf = async () => {
    if (!analysis) return;
    const res = await fetch(`${backend}/api/report/pdf/${analysis.analysisId}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${symbol}-report.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const charts = useMemo(() => {
    if (!analysis?.charts) return [];
    return Object.entries(analysis.charts).map(([k, v]) => ({ key: k, url: v }));
  }, [analysis]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      <BrandHeader />

      <div className="card p-4 space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="text-sm text-gray-600">{t("controls.search.placeholder")}</label>
            <input
              value={symbol}
              onChange={(e) => {
                setSymbol(e.target.value.toUpperCase());
                setQuery(e.target.value);
              }}
              className="w-full border rounded px-3 py-2"
              list="symbol-options"
            />
            <datalist id="symbol-options">
              {symbols.map((s: any) => (
                <option key={s.symbol} value={s.symbol}>
                  {s.name}
                </option>
              ))}
            </datalist>
          </div>
          <div>
            <label className="text-sm text-gray-600">{t("controls.specific")}</label>
            <select value={mode} onChange={(e) => setMode(e.target.value as any)} className="w-full border rounded px-3 py-2">
              <option value="latest">{t("controls.latest")}</option>
              <option value="specific">{t("controls.specific")}</option>
            </select>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-sm text-gray-600">{t("controls.year")}</label>
              <input
                type="number"
                value={year ?? ""}
                onChange={(e) => setYear(e.target.value ? Number(e.target.value) : undefined)}
                className="w-full border rounded px-3 py-2"
              />
            </div>
            <div>
              <label className="text-sm text-gray-600">{t("controls.quarter")}</label>
              <input
                type="number"
                min={1}
                max={4}
                value={quarter ?? ""}
                onChange={(e) => setQuarter(e.target.value ? Number(e.target.value) : undefined)}
                className="w-full border rounded px-3 py-2"
              />
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={generate} disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded">
            {loading ? "..." : t("actions.generate")}
          </button>
          <button onClick={downloadPdf} disabled={!analysis} className="px-3 py-2 rounded border">
            {t("actions.downloadPdf")}
          </button>
        </div>
      </div>

      {analysis && (
        <>
          <div className="card p-4 space-y-2 sticky top-0 z-10">
            <h2 className="text-xl font-semibold">{analysis.summary?.title || t("summary.title")}</h2>
            <ul className="list-disc pl-5 space-y-1">
              {analysis.summary?.bullets?.map((b: string, i: number) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </div>

          <div className="card p-4 space-y-3">
            <h3 className="text-lg font-semibold">{t("nav.overview")}</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(analysis.kpis || {}).map(([k, v]) => (
                <div key={k} className="p-3 rounded border">
                  <div className="text-xs uppercase text-gray-500">{k}</div>
                  <div className="text-lg font-semibold tabular-nums">{typeof v === "number" ? v.toFixed(2) : String(v)}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {charts.map((c) => (
              <div key={c.key} className="card p-2">
                <div className="text-sm font-semibold px-2 py-1">{c.key}</div>
                <img src={c.url} alt={c.key} className="w-full" />
              </div>
            ))}
          </div>

          <div className="card p-4">
            <h3 className="text-lg font-semibold mb-2">{t("nav.transcript")}</h3>
            <ul className="list-disc pl-5 space-y-1">
              {analysis.transcriptHighlights.map((h, i) => (
                <li key={i}>{h}</li>
              ))}
            </ul>
          </div>

          {analysis.graphEnabled && (
            <div className="card p-4">
              <h3 className="text-lg font-semibold mb-2">{t("graph.title")}</h3>
              <div className="h-80">
                <ForceGraph2D graphData={graphData} nodeLabel={(n: any) => n.id} nodeAutoColorBy="label" />
              </div>
            </div>
          )}

          <div className="card p-4">
            <h3 className="text-lg font-semibold mb-2">{t("data.title")}</h3>
            <pre className="text-xs overflow-auto max-h-96">{JSON.stringify(analysis.tables, null, 2)}</pre>
          </div>
        </>
      )}
    </div>
  );
}
EOF2

cat > components/StickySummary.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";

export default function StickySummary({ title, bullets }: { title: string; bullets: string[] }) {
  const { t } = useI18n();
  return (
    <div className="card p-4 space-y-2 sticky top-0 z-10">
      <h2 className="text-xl font-semibold">{title || t("summary.title")}</h2>
      <ul className="list-disc pl-5 space-y-1">
        {bullets?.map((b, i) => (
          <li key={i}>{b}</li>
        ))}
      </ul>
    </div>
  );
}
EOF2

cat > components/ControlBar.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";

type Props = {
  symbol: string;
  setSymbol: (s: string) => void;
  year?: number;
  setYear: (n?: number) => void;
  quarter?: number;
  setQuarter: (n?: number) => void;
  mode: "latest" | "specific";
  setMode: (m: "latest" | "specific") => void;
  onGenerate: () => void;
};

export default function ControlBar(props: Props) {
  const { t } = useI18n();
  const { symbol, setSymbol, year, setYear, quarter, setQuarter, mode, setMode, onGenerate } = props;
  return (
    <div className="card p-4 space-y-3">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div>
          <label className="text-sm text-gray-600">{t("controls.search.placeholder")}</label>
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="text-sm text-gray-600">{t("controls.specific")}</label>
          <select value={mode} onChange={(e) => setMode(e.target.value as any)} className="w-full border rounded px-3 py-2">
            <option value="latest">{t("controls.latest")}</option>
            <option value="specific">{t("controls.specific")}</option>
          </select>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-sm text-gray-600">{t("controls.year")}</label>
            <input
              type="number"
              value={year ?? ""}
              onChange={(e) => setYear(e.target.value ? Number(e.target.value) : undefined)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="text-sm text-gray-600">{t("controls.quarter")}</label>
            <input
              type="number"
              min={1}
              max={4}
              value={quarter ?? ""}
              onChange={(e) => setQuarter(e.target.value ? Number(e.target.value) : undefined)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
        </div>
      </div>
      <button onClick={onGenerate} className="px-4 py-2 bg-blue-600 text-white rounded">
        {t("actions.generate")}
      </button>
    </div>
  );
}
EOF2

cat > components/SymbolSearch.tsx <<'EOF2'
"use client";
import { useEffect, useState } from "react";
import { useI18n } from "../hooks/useI18n";
const backend = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function SymbolSearch({ symbol, setSymbol }: { symbol: string; setSymbol: (s: string) => void }) {
  const { t } = useI18n();
  const [symbols, setSymbols] = useState<any[]>([]);
  const [query, setQuery] = useState("");
  useEffect(() => {
    if (query.length < 1) return;
    const tmr = setTimeout(async () => {
      const res = await fetch(`${backend}/api/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      setSymbols(data || []);
    }, 250);
    return () => clearTimeout(tmr);
  }, [query]);
  return (
    <div>
      <label className="text-sm text-gray-600">{t("controls.search.placeholder")}</label>
      <input
        value={symbol}
        onChange={(e) => {
          setSymbol(e.target.value.toUpperCase());
          setQuery(e.target.value);
        }}
        className="w-full border rounded px-3 py-2"
        list="symbol-options"
      />
      <datalist id="symbol-options">
        {symbols.map((s: any) => (
          <option key={s.symbol} value={s.symbol}>
            {s.name}
          </option>
        ))}
      </datalist>
    </div>
  );
}
EOF2

cat > components/QuarterPicker.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";

export default function QuarterPicker({
  year,
  setYear,
  quarter,
  setQuarter,
}: {
  year?: number;
  setYear: (n?: number) => void;
  quarter?: number;
  setQuarter: (n?: number) => void;
}) {
  const { t } = useI18n();
  return (
    <div className="grid grid-cols-2 gap-2">
      <div>
        <label className="text-sm text-gray-600">{t("controls.year")}</label>
        <input
          type="number"
          value={year ?? ""}
          onChange={(e) => setYear(e.target.value ? Number(e.target.value) : undefined)}
          className="w-full border rounded px-3 py-2"
        />
      </div>
      <div>
        <label className="text-sm text-gray-600">{t("controls.quarter")}</label>
        <input
          type="number"
          min={1}
          max={4}
          value={quarter ?? ""}
          onChange={(e) => setQuarter(e.target.value ? Number(e.target.value) : undefined)}
          className="w-full border rounded px-3 py-2"
        />
      </div>
    </div>
  );
}
EOF2

cat > components/PeersSelect.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";

export default function PeersSelect({ peers, setPeers }: { peers: string[]; setPeers: (p: string[]) => void }) {
  const { t } = useI18n();
  const addPeer = () => setPeers([...(peers || []), ""]);
  return (
    <div>
      <label className="text-sm text-gray-600">{t("peers.title")}</label>
      <div className="space-y-2">
        {(peers || []).map((p, i) => (
          <input
            key={i}
            value={p}
            onChange={(e) => {
              const clone = [...peers];
              clone[i] = e.target.value.toUpperCase();
              setPeers(clone);
            }}
            className="w-full border rounded px-3 py-2"
          />
        ))}
        <button type="button" onClick={addPeer} className="text-sm text-blue-600 underline">
          {t("actions.addPeers")}
        </button>
      </div>
    </div>
  );
}
EOF2

cat > components/KpiGrid.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";
import { useFormat } from "../lib/format";

export default function KpiGrid({ kpis }: { kpis: Record<string, any> }) {
  const { t } = useI18n();
  const { formatNumber } = useFormat();
  if (!kpis) return null;
  const items: { key: keyof typeof kpis; label: string; style?: string }[] = [
    { key: "revenue", label: t("kpi.revenue") },
    { key: "epsDiluted", label: t("kpi.eps") },
    { key: "grossMarginPct", label: t("kpi.grossMargin") },
    { key: "operatingMarginPct", label: t("kpi.opMargin") },
    { key: "netMarginPct", label: t("kpi.netMargin") },
    { key: "cfo", label: t("kpi.cfo") },
    { key: "capex", label: t("kpi.capex") },
    { key: "fcf", label: t("kpi.fcf") },
    { key: "fcfMarginPct", label: t("kpi.fcfMargin") },
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      {items.map((item) => (
        <div key={item.key} className="p-3 rounded border">
          <div className="text-xs uppercase text-gray-500">{item.label}</div>
          <div className="text-lg font-semibold font-tabular">
            {typeof kpis[item.key] === "number" ? formatNumber(Number(kpis[item.key]), "decimal") : kpis[item.key] ?? "-"}
          </div>
        </div>
      ))}
    </div>
  );
}
EOF2

cat > components/EventWindow.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";
import { useFormat } from "../lib/format";

export default function EventWindow({ events }: { events: any[] }) {
  const { t } = useI18n();
  const { formatSignedPercent } = useFormat();
  if (!events || !events.length) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left">
          <th className="py-1">{t("event.title")}</th>
          <th className="py-1">{t("event.shock")}</th>
        </tr>
      </thead>
      <tbody>
        {events.map((e, i) => (
          <tr key={i} className="border-t">
            <td className="py-1">{e.window}</td>
            <td className="py-1">
              {formatSignedPercent(e.returnPct || 0)} {e.shock ? `(${t("event.shock")})` : `(${t("event.noShock")})`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
EOF2

cat > components/SectionTabs.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";
import clsx from "clsx";

export default function SectionTabs({ active, setActive }: { active: string; setActive: (k: string) => void }) {
  const { t } = useI18n();
  const tabs = [
    { key: "overview", label: t("nav.overview") },
    { key: "financials", label: t("nav.financials") },
    { key: "transcript", label: t("nav.transcript") },
    { key: "peers", label: t("nav.peers") },
    { key: "graph", label: t("nav.graph") },
    { key: "data", label: t("nav.data") },
  ];
  return (
    <div role="tablist" className="flex gap-2 border-b overflow-x-auto">
      {tabs.map((tab) => (
        <button
          role="tab"
          key={tab.key}
          aria-pressed={active === tab.key}
          onClick={() => setActive(tab.key)}
          className={clsx("px-3 py-2 text-sm", active === tab.key ? "border-b-2 border-blue-600 font-semibold" : "text-gray-500")}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
EOF2

cat > components/ChartStrip.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";

export default function ChartStrip({ charts }: { charts: Record<string, string> }) {
  const { t } = useI18n();
  if (!charts) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {Object.entries(charts).map(([k, v]) => (
        <div key={k} className="card p-2">
          <div className="text-sm font-semibold px-2 py-1">{k}</div>
          <img src={v} alt={k} className="w-full" />
        </div>
      ))}
    </div>
  );
}
EOF2

cat > components/PeersBlock.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";

export default function PeersBlock({ peers }: { peers: any }) {
  const { t } = useI18n();
  if (!peers) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <div className="card p-4 space-y-2">
      <h3 className="text-lg font-semibold">{t("peers.title")}</h3>
      <div className="text-sm text-gray-600">{t("peers.median")}: {JSON.stringify(peers.medians)}</div>
      <div className="text-sm text-gray-600">{t("peers.vsCompany")}: {peers.peers?.join(", ")}</div>
    </div>
  );
}
EOF2

cat > components/GraphPreview.tsx <<'EOF2'
"use client";
import dynamic from "next/dynamic";
import { useI18n } from "../hooks/useI18n";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function GraphPreview({ enabled, data }: { enabled: boolean; data: { nodes: any[]; links: any[] } }) {
  const { t } = useI18n();
  if (!enabled) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <div className="card p-4">
      <h3 className="text-lg font-semibold mb-2">{t("graph.title")}</h3>
      <div className="h-80">
        <ForceGraph2D graphData={data} nodeLabel={(n: any) => n.id} nodeAutoColorBy="label" />
      </div>
    </div>
  );
}
EOF2

cat > components/DataTables.tsx <<'EOF2'
"use client";
import { useI18n } from "../hooks/useI18n";

export default function DataTables({ tables }: { tables: any }) {
  const { t } = useI18n();
  if (!tables) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <div className="card p-4">
      <h3 className="text-lg font-semibold mb-2">{t("data.title")}</h3>
      <pre className="text-xs overflow-auto max-h-96">{JSON.stringify(tables, null, 2)}</pre>
    </div>
  );
}
EOF2

cat > app/globals.css <<'EOF2'
@tailwind base;
@tailwind components;
@tailwind utilities;
:root { color-scheme: light; }
body { @apply bg-gray-50 text-gray-900; }
.card { @apply rounded-lg border bg-white shadow-sm; }
.font-tabular { font-variant-numeric: tabular-nums; }
EOF2

npm install
npm run dev -- -p 3000
