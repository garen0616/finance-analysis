#!/usr/bin/env bash
set -euo pipefail

# Locate frontend dir
if [ -d "repo/frontend" ]; then
  FRONTEND_DIR="repo/frontend"
elif [ -d "frontend" ]; then
  FRONTEND_DIR="frontend"
else
  echo "frontend directory not found" >&2
  exit 1
fi

# Check node version
node -v >/dev/null 2>&1 || { echo "Node.js >=18 required" >&2; exit 1; }
node - <<'JS'
const v = process.versions.node.split('.').map(Number);
if (v[0] < 18) { console.error('Node.js >=18 required'); process.exit(1); }
JS

cd "$FRONTEND_DIR"

npm install framer-motion clsx tailwind-merge lucide-react swr react-plotly.js plotly.js-basic-dist-min react-force-graph-2d tailwindcss-animate

cat > tailwind.config.ts <<'EOF2'
import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1.25rem" },
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"] },
      colors: {
        surface: {
          DEFAULT: "#f8fafc",
          muted: "#f1f5f9",
          line: "#e6e8ec",
          glass: "#ffffffb3",
        },
        accent: {
          start: "#6366f1",
          end: "#06b6d4",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
        },
      },
      boxShadow: {
        glass: "0 1px 0 0 rgba(0,0,0,0.04), 0 8px 20px -12px rgba(0,0,0,0.25)",
      },
      keyframes: {
        "grid-move": { "0%": { transform: "translateY(0)" }, "100%": { transform: "translateY(-8px)" } },
        fade: { "0%": { opacity: 0 }, "100%": { opacity: 1 } },
        slide: { "0%": { opacity: 0, transform: "translateY(6px)" }, "100%": { opacity: 1, transform: "translateY(0)" } },
      },
      animation: {
        "grid-move": "grid-move 14s ease-in-out infinite alternate",
        fade: "fade 200ms ease-out",
        slide: "slide 240ms ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
EOF2

cat > app/globals.css <<'EOF2'
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg: #f8fafc;
  --fg: #0f172a;
  --line: #e6e8ec;
  --accent: linear-gradient(90deg, #6366f1, #06b6d4);
}

body {
  @apply bg-[var(--bg)] text-[var(--fg)] font-sans antialiased;
}

.glass { @apply backdrop-blur bg-white/70 shadow-glass border border-[var(--line)]; }
.font-tabular { font-variant-numeric: tabular-nums; }

.animated-grid { position: relative; isolation: isolate; }
.animated-grid::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(to right, rgba(99,102,241,0.12) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(99,102,241,0.12) 1px, transparent 1px);
  background-size: 120px 120px;
  mask-image: radial-gradient(ellipse at center, rgba(0,0,0,0.4), transparent 65%);
  opacity: 0.6;
}
@media (prefers-reduced-motion: no-preference) {
  .animated-grid::before { animation: grid-move 16s ease-in-out infinite alternate; }
}

.gradient-bar { background-image: var(--accent); height: 3px; border-radius: 9999px; }
EOF2

cat > app/layout.tsx <<'EOF2'
import "./globals.css";
import type { ReactNode } from "react";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"], display: "swap" });

export const metadata = {
  title: "Finance Analytics",
  description: "Earnings analytics dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-[var(--bg)] text-[var(--fg)]`}>
        <div className="gradient-bar" />
        {children}
      </body>
    </html>
  );
}
EOF2

cat > lib/api.ts <<'EOF2'
const base = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function getJSON<T>(path: string) {
  const res = await fetch(`${base}${path}`);
  if (!res.ok) throw new Error(`GET ${path} ${res.status}`);
  return res.json() as Promise<T>;
}

export async function postJSON<T>(path: string, body: any) {
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} ${res.status}`);
  return res.json() as Promise<T>;
}

export function pdfUrl(analysisId: string) { return `${base}/api/report/pdf/${analysisId}`; }
export const apiBase = base;
EOF2

mkdir -p components/ui
cat > components/ui/cn.ts <<'EOF2'
import { clsx, ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
export function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }
EOF2

cat > components/ui/button.tsx <<'EOF2'
"use client";
import { cn } from "./cn";
import { forwardRef } from "react";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "outline"; size?: "sm" | "md" };

export const Button = forwardRef<HTMLButtonElement, Props>(function Button({ className, variant = "primary", size = "md", ...props }, ref) {
  const base = "inline-flex items-center justify-center rounded-md font-medium transition-colors";
  const variants = {
    primary: "bg-gradient-to-r from-accent-start to-accent-end text-white shadow-sm hover:opacity-90",
    ghost: "bg-transparent hover:bg-surface-muted text-[var(--fg)]",
    outline: "border border-[var(--line)] bg-white/70 hover:bg-surface-muted",
  } as const;
  const sizes = { sm: "px-3 py-1.5 text-sm", md: "px-4 py-2 text-sm" } as const;
  return <button ref={ref} className={cn(base, variants[variant], sizes[size], className)} {...props} />;
});
EOF2

cat > components/ui/card.tsx <<'EOF2'
import { cn } from "./cn";
export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("glass rounded-md p-4", className)}>{children}</div>;
}
EOF2

cat > components/ui/tabs.tsx <<'EOF2'
"use client";
import { useId } from "react";
import { cn } from "./cn";

type Tab = { key: string; label: string };
export function Tabs({ tabs, active, onChange }: { tabs: Tab[]; active: string; onChange: (k: string) => void }) {
  const id = useId();
  return (
    <div role="tablist" aria-label="sections" className="flex gap-2 border-b border-[var(--line)] overflow-x-auto">
      {tabs.map((t) => (
        <button
          key={t.key}
          id={`${id}-${t.key}`}
          role="tab"
          aria-selected={active === t.key}
          onClick={() => onChange(t.key)}
          className={cn("px-3 py-2 text-sm transition-colors", active === t.key ? "border-b-2 border-accent-600 text-accent-600" : "text-slate-500")}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
EOF2

cat > components/ui/tooltip.tsx <<'EOF2'
"use client";
import { useState } from "react";
import { cn } from "./cn";
export function Tooltip({ label, children }: { label: string; children: React.ReactNode }) {
  const [hover, setHover] = useState(false);
  return (
    <span className="relative inline-flex"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
      aria-label={label}
    >
      {children}
      {hover && (
        <span className={cn("absolute z-50 whitespace-nowrap rounded-md bg-slate-900 text-white text-xs px-2 py-1 shadow", "translate-y-[-8px] left-1/2 -translate-x-1/2")}>{label}</span>
      )}
    </span>
  );
}
EOF2

cat > components/ui/skeleton.tsx <<'EOF2'
import { cn } from "./cn";
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-slate-200", className)} />;
}
EOF2

cat > components/ui/separator.tsx <<'EOF2'
export function Separator() { return <div className="h-px w-full bg-[var(--line)]" />; }
EOF2

cat > components/ui/badge.tsx <<'EOF2'
import { cn } from "./cn";
export function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "positive" | "negative" }) {
  const styles = {
    neutral: "bg-slate-100 text-slate-700",
    positive: "bg-emerald-100 text-emerald-700",
    negative: "bg-rose-100 text-rose-700",
  }[tone];
  return <span className={cn("px-2 py-1 rounded-full text-xs font-medium", styles)}>{children}</span>;
}
EOF2

cat > components/ui/ToggleThemeButton.tsx <<'EOF2'
"use client";
import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "./button";

export function ToggleThemeButton() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    setDark(media.matches);
  }, []);
  useEffect(() => { document.documentElement.classList.toggle("dark", dark); }, [dark]);
  return (
    <Button variant="ghost" size="sm" onClick={() => setDark((d) => !d)} aria-label="Toggle theme">
      {dark ? <Sun size={16} /> : <Moon size={16} />}
    </Button>
  );
}
EOF2

mkdir -p components/brand components/controls
cat > components/brand/Header.tsx <<'EOF2'
"use client";
import { Button } from "../ui/button";
import { ToggleThemeButton } from "../ui/ToggleThemeButton";
import { Download, FileDown } from "lucide-react";
import { pdfUrl } from "../../lib/api";

export default function Header({ analysisId }: { analysisId?: string }) {
  return (
    <div className="flex items-center justify-between py-4">
      <div className="flex items-center gap-3">
        <img src="/brand.svg" alt="brand" className="h-8 w-8" />
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500">Finance Analytics</div>
          <div className="text-xl font-semibold">Earnings Insight</div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <ToggleThemeButton />
        <Button variant="outline" size="sm" disabled={!analysisId} onClick={() => analysisId && window.open(pdfUrl(analysisId), "_blank")}> <Download className="mr-2 h-4 w-4" /> PDF </Button>
        <Button variant="ghost" size="sm"> <FileDown className="mr-2 h-4 w-4" /> CSV </Button>
      </div>
    </div>
  );
}
EOF2

cat > components/brand/StickySummary.tsx <<'EOF2'
"use client";
import { Badge } from "../ui/badge";

export default function StickySummary({ bullets }: { bullets: string[] }) {
  if (!bullets?.length) return null;
  return (
    <div className="sticky top-0 z-50 glass rounded-md p-4 border-b-2 border-gradient-to-r from-accent-500 to-accent-end shadow-md backdrop-blur">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Executive Summary</div>
      <div className="flex flex-wrap gap-2">
        {bullets.slice(0, 6).map((b, i) => (
          <Badge key={i} tone="neutral">{b}</Badge>
        ))}
      </div>
    </div>
  );
}
EOF2

cat > components/brand/ControlBar.tsx <<'EOF2'
"use client";
import { useState } from "react";
import SymbolSearch from "../controls/SymbolSearch";
import QuarterPicker from "../controls/QuarterPicker";
import { Button } from "../ui/button";

export default function ControlBar({ symbol, setSymbol, onGenerate }: { symbol: string; setSymbol: (s: string) => void; onGenerate: (mode: "latest" | "specific", year?: number, quarter?: number) => void; }) {
  const [mode, setMode] = useState<"latest" | "specific">("latest");
  const [year, setYear] = useState<number | undefined>(undefined);
  const [quarter, setQuarter] = useState<number | undefined>(undefined);
  return (
    <div className="glass p-4 rounded-md space-y-4">
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1"><SymbolSearch symbol={symbol} setSymbol={setSymbol} /></div>
        <div className="flex-1"><QuarterPicker mode={mode} setMode={setMode} year={year} setYear={setYear} quarter={quarter} setQuarter={setQuarter} /></div>
        <div className="flex items-end"><Button onClick={() => onGenerate(mode, year, quarter)}>Generate</Button></div>
      </div>
    </div>
  );
}
EOF2

cat > components/brand/KpiCard.tsx <<'EOF2'
"use client";
import { Badge } from "../ui/badge";

export type KpiCardProps = { label: string; value?: number | string; delta?: number; spark?: number[] };

export default function KpiCard({ label, value = "-", delta, spark = [] }: KpiCardProps) {
  const tone = delta == null ? "neutral" : delta > 0 ? "positive" : delta < 0 ? "negative" : "neutral";
  const deltaDisplay = delta == null ? "" : `${delta > 0 ? "+" : ""}${delta.toFixed(2)}%`;
  return (
    <div className="glass rounded-md p-3 flex flex-col gap-2">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-2xl font-semibold font-tabular">{typeof value === "number" ? value.toLocaleString() : value}</div>
      <div className="flex items-center justify-between text-xs text-slate-500">
        <Badge tone={tone as any}>{deltaDisplay || "—"}</Badge>
        <SparkLine data={spark} />
      </div>
    </div>
  );
}

function SparkLine({ data }: { data: number[] }) {
  if (!data?.length) return <div className="text-slate-400">—</div>;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const points = data.map((d, i) => {
    const x = (i / Math.max(1, data.length - 1)) * 100;
    const y = max === min ? 50 : 100 - ((d - min) / (max - min)) * 100;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg viewBox="0 0 100 100" className="w-20 h-10">
      <defs>
        <linearGradient id="spark" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#06b6d4" />
        </linearGradient>
      </defs>
      <polyline fill="none" stroke="url(#spark)" strokeWidth="3" points={points} strokeLinecap="round" />
    </svg>
  );
}
EOF2

cat > components/brand/ChartCard.tsx <<'EOF2'
"use client";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function ChartCard({ title, desc, dataUrl }: { title: string; desc?: string; dataUrl?: string }) {
  return (
    <div className="glass rounded-md p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold">{title}</div>
          {desc && <div className="text-xs text-slate-500">{desc}</div>}
        </div>
      </div>
      {dataUrl ? (
        <img src={dataUrl} alt={title} className="w-full h-auto rounded" />
      ) : (
        <div className="text-sm text-slate-500">No chart data</div>
      )}
    </div>
  );
}
EOF2

cat > components/brand/DataTable.tsx <<'EOF2'
"use client";
import { Button } from "../ui/button";

export default function DataTable({ title, rows }: { title: string; rows: any[] }) {
  return (
    <div className="glass rounded-md p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold">{title}</div>
        <Button variant="outline" size="sm">CSV</Button>
      </div>
      <div className="overflow-auto max-h-96">
        <table className="min-w-full text-sm">
          <thead className="sticky top-0 bg-white/90 backdrop-blur">
            <tr>
              {rows?.[0] ? Object.keys(rows[0]).map((k) => <th key={k} className="text-left px-3 py-2 text-slate-500">{k}</th>) : <th>—</th>}
            </tr>
          </thead>
          <tbody>
            {rows?.length ? rows.map((r, i) => (
              <tr key={i} className="border-t border-[var(--line)]">
                {Object.values(r).map((v, j) => (
                  <td key={j} className="px-3 py-2 whitespace-nowrap">{typeof v === "number" ? v.toLocaleString() : String(v ?? "")}</td>
                ))}
              </tr>
            )) : (
              <tr><td className="px-3 py-4 text-slate-500">No data</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
EOF2

cat > components/brand/PeersPanel.tsx <<'EOF2'
"use client";
import { useState } from "react";
import { Badge } from "../ui/badge";

export default function PeersPanel({ peers }: { peers: { peers?: any[]; medians?: any } }) {
  const list = peers?.peers || [];
  const [selected, setSelected] = useState<string | null>(null);
  return (
    <div className="glass rounded-md p-4 space-y-3">
      <div className="text-lg font-semibold">Peers</div>
      <div className="flex flex-wrap gap-2">
        {list.map((p: any) => (
          <Badge key={p.symbol || p} tone={selected === (p.symbol || p) ? "positive" : "neutral"}>{p.symbol || p}</Badge>
        ))}
      </div>
      <div className="text-sm text-slate-500">Medians: {JSON.stringify(peers?.medians || {})}</div>
    </div>
  );
}
EOF2

cat > components/brand/GraphPreview.tsx <<'EOF2'
"use client";
import dynamic from "next/dynamic";
import { useState } from "react";
import { Button } from "../ui/button";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function GraphPreview({ enabled, data }: { enabled?: boolean; data: { nodes: any[]; links: any[] } }) {
  const [open, setOpen] = useState(true);
  if (!enabled) return null;
  return (
    <div className="glass rounded-md p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold">Graph Preview</div>
        <Button variant="ghost" size="sm" onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"}</Button>
      </div>
      {open && (
        <div className="h-80">
          <ForceGraph2D graphData={data} nodeLabel={(n: any) => n.id} nodeAutoColorBy="label" />
        </div>
      )}
    </div>
  );
}
EOF2

cat > components/brand/EmptyState.tsx <<'EOF2'
export function EmptyState({ message = "No data available" }: { message?: string }) {
  return <div className="text-sm text-slate-500">{message}</div>;
}
EOF2

cat > components/brand/ErrorState.tsx <<'EOF2'
"use client";
import { Button } from "../ui/button";
export function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="glass rounded-md p-4 flex items-center justify-between">
      <div className="text-sm text-rose-600">Something went wrong. Please retry.</div>
      {onRetry && <Button variant="outline" size="sm" onClick={onRetry}>Retry</Button>}
    </div>
  );
}
EOF2

cat > components/controls/SymbolSearch.tsx <<'EOF2'
"use client";
import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { apiBase } from "../../lib/api";
import { cn } from "../ui/cn";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function SymbolSearch({ symbol, setSymbol }: { symbol: string; setSymbol: (s: string) => void }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const { data } = useSWR(query.length > 0 ? `${apiBase}/api/search?q=${encodeURIComponent(query)}` : null, fetcher, { dedupingInterval: 400 });
  const list = useMemo(() => data || [], [data]);
  return (
    <div className="relative">
      <label className="text-xs uppercase text-slate-500">Symbol</label>
      <input
        value={symbol}
        onChange={(e) => { setSymbol(e.target.value.toUpperCase()); setQuery(e.target.value); setOpen(true); }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        className="w-full mt-1 px-3 py-2 rounded-md border border-[var(--line)] bg-white/80"
        placeholder="Search or type symbol"
      />
      {open && list.length > 0 && (
        <div className="absolute mt-1 w-full glass rounded-md max-h-64 overflow-auto z-40">
          {list.map((item: any) => (
            <div
              key={item.symbol}
              className={cn("px-3 py-2 cursor-pointer hover:bg-surface-muted")}
              onMouseDown={() => { setSymbol(item.symbol); setOpen(false); }}
            >
              <div className="font-medium">{item.symbol}</div>
              <div className="text-xs text-slate-500">{item.name}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
EOF2

cat > components/controls/QuarterPicker.tsx <<'EOF2'
"use client";
import { cn } from "../ui/cn";

export default function QuarterPicker({
  mode,
  setMode,
  year,
  setYear,
  quarter,
  setQuarter,
}: {
  mode: "latest" | "specific";
  setMode: (m: "latest" | "specific") => void;
  year?: number;
  setYear: (n?: number) => void;
  quarter?: number;
  setQuarter: (n?: number) => void;
}) {
  const options = ["latest", "specific"] as const;
  return (
    <div className="space-y-2">
      <div className="text-xs uppercase text-slate-500">Mode</div>
      <div className="inline-flex rounded-md border border-[var(--line)] overflow-hidden">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => setMode(opt)}
            className={cn("px-3 py-2 text-sm", mode === opt ? "bg-surface-muted font-medium" : "bg-white/70")}
          >
            {opt === "latest" ? "Latest" : "Specific"}
          </button>
        ))}
      </div>
      {mode === "specific" && (
        <div className="flex gap-2">
          <input
            type="number"
            placeholder="Year"
            value={year ?? ""}
            onChange={(e) => setYear(e.target.value ? Number(e.target.value) : undefined)}
            className="w-24 px-3 py-2 rounded-md border border-[var(--line)]"
          />
          <select
            value={quarter ?? ""}
            onChange={(e) => setQuarter(e.target.value ? Number(e.target.value) : undefined)}
            className="px-3 py-2 rounded-md border border-[var(--line)]"
          >
            <option value="">Quarter</option>
            {[1, 2, 3, 4].map((q) => (
              <option key={q} value={q}>{`Q${q}`}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
EOF2

cat > app/page.tsx <<'EOF2'
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
EOF2

mkdir -p public
cat > public/brand.svg <<'EOF2'
<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop stop-color="#6366f1" offset="0%"/>
      <stop stop-color="#06b6d4" offset="100%"/>
    </linearGradient>
  </defs>
  <rect x="10" y="10" width="100" height="100" rx="20" fill="url(#g)" opacity="0.15"/>
  <path d="M35 80 L55 45 L70 70 L85 40" stroke="url(#g)" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="35" cy="80" r="6" fill="#6366f1"/>
  <circle cx="55" cy="45" r="6" fill="#6366f1"/>
  <circle cx="70" cy="70" r="6" fill="#06b6d4"/>
  <circle cx="85" cy="40" r="6" fill="#06b6d4"/>
</svg>
EOF2

cat > README.UI.md <<'EOF2'
# UI Theme Notes

- Palette: slate neutrals with indigo→cyan accent (`app/globals.css`).
- Components: see `components/brand` and `components/ui`.
- APIs: uses `NEXT_PUBLIC_BACKEND_URL` for `/api/search`, `/api/analyze`, `/api/report/pdf/:id`.
- Layout: `app/page.tsx` with tabs for Overview, Financials, Transcript, Peers, Graph, Data.
- Animations respect `prefers-reduced-motion`.
- Tweak tokens in `tailwind.config.ts` and CSS variables in `app/globals.css`.
EOF2

# kill existing next on 3000 if any
lsof -ti:3000 | xargs -r kill || true

npm install
npm run dev
