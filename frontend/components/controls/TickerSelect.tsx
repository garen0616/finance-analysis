"use client";
import useSWR from "swr";
import { apiBase } from "../../lib/api";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function TickerSelect({ dataset, value, onChange }: { dataset?: string; value: string; onChange: (v: string) => void }) {
  const { data, isLoading } = useSWR(dataset ? `${apiBase}/api/dataset/tickers?dataset=${encodeURIComponent(dataset)}` : null, fetcher);
  const list = data || [];
  return (
    <div className="space-y-2">
      <label className="text-xs uppercase tracking-[0.3em] text-slate-400">Ticker</label>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value.toUpperCase())}
        disabled={!dataset}
        className="w-full glass-input disabled:opacity-50"
      >
        <option value="">{isLoading ? "Loading..." : "Select"}</option>
        {list.map((t: any) => (
          <option key={t.symbol} value={t.symbol} className="bg-night-900 text-slate-200">{t.symbol} ({t.count})</option>
        ))}
      </select>
    </div>
  );
}
