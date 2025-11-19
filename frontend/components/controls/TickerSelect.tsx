"use client";
import useSWR from "swr";
import { apiBase } from "../../lib/api";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function TickerSelect({ dataset, value, onChange }: { dataset?: string; value: string; onChange: (v: string) => void }) {
  const { data, isLoading } = useSWR(dataset ? `${apiBase}/api/dataset/tickers?dataset=${encodeURIComponent(dataset)}` : null, fetcher);
  const list = data || [];
  return (
    <div>
      <label className="text-xs uppercase text-slate-500">Ticker</label>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value.toUpperCase())}
        disabled={!dataset}
        className="w-full mt-1 px-3 py-2 rounded-md border border-[var(--line)] bg-white/80 disabled:opacity-60"
      >
        <option value="">{isLoading ? "Loading..." : "Select"}</option>
        {list.map((t: any) => (
          <option key={t.symbol} value={t.symbol}>{t.symbol} ({t.count})</option>
        ))}
      </select>
    </div>
  );
}
