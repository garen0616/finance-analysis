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
