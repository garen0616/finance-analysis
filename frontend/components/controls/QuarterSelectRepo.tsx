"use client";
import useSWR from "swr";
import { apiBase } from "../../lib/api";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

type Period = { fiscalYear?: number; fiscalQuarter?: number; periodEnd?: string };

export default function QuarterSelectRepo({
  dataset,
  symbol,
  year,
  quarter,
  onChange,
}: {
  dataset?: string;
  symbol?: string;
  year?: number;
  quarter?: number;
  onChange: (y?: number, q?: number) => void;
}) {
  const { data, isLoading } = useSWR(
    dataset && symbol ? `${apiBase}/api/dataset/periods?dataset=${encodeURIComponent(dataset)}&symbol=${encodeURIComponent(symbol)}` : null,
    fetcher
  );
  const options: Period[] = data || [];
  const value = `${year || ""}-${quarter || ""}`;

  return (
    <div className="space-y-2">
      <label className="text-xs uppercase tracking-[0.3em] text-slate-400">Period</label>
      <select
        value={value}
        onChange={(e) => {
          const [y, q] = e.target.value.split("-").map((v) => (v ? Number(v) : undefined));
          onChange(y, q);
        }}
        disabled={!dataset || !symbol}
        className="w-full glass-input disabled:opacity-50"
      >
        <option value="">{isLoading ? "Loading..." : "Select period"}</option>
        {options.map((p, i) => {
          const key = `${p.fiscalYear || "?"}-Q${p.fiscalQuarter || "?"}-${i}`;
          const val = `${p.fiscalYear || ""}-${p.fiscalQuarter || ""}`;
          const label = `${p.fiscalYear || "?"} Q${p.fiscalQuarter || "?"}${p.periodEnd ? ` • ${p.periodEnd}` : ""}`;
          return (
            <option key={key} value={val} className="bg-night-900 text-slate-200">
              {label}
            </option>
          );
        })}
      </select>
    </div>
  );
}
