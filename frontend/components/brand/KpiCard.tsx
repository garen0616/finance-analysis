"use client";
import { Badge } from "../ui/badge";

export type KpiCardProps = { label: string; value?: number | string; delta?: number; spark?: number[] };

export default function KpiCard({ label, value = "-", delta, spark = [] }: KpiCardProps) {
  const tone = delta == null ? "neutral" : delta > 0 ? "positive" : delta < 0 ? "negative" : "neutral";
  const deltaDisplay = delta == null ? "" : `${delta > 0 ? "+" : ""}${delta.toFixed(2)}%`;
  return (
    <div className="glass-panel rounded-3xl p-4 flex flex-col gap-3">
      <div className="text-xs uppercase tracking-[0.4em] text-slate-400">{label}</div>
      <div className="text-3xl font-semibold font-tabular text-white">
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
      <div className="flex items-center justify-between text-xs text-slate-300">
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
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#2563eb" />
        </linearGradient>
      </defs>
      <polyline fill="none" stroke="url(#spark)" strokeWidth="3" points={points} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
