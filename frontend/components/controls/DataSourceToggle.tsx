"use client";
import { cn } from "../ui/cn";
export default function DataSourceToggle({ value, onChange }: { value: "repo" | "fmp"; onChange: (v: "repo" | "fmp") => void }) {
  const opts = [
    { value: "repo", label: "GitHub Sample" },
    { value: "fmp", label: "FMP Live" },
  ];
  return (
    <div>
      <div className="text-xs uppercase text-slate-500">Data Source</div>
      <div className="inline-flex rounded-md border border-[var(--line)] overflow-hidden mt-1">
        {opts.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value as any)}
            className={cn("px-3 py-2 text-sm", value === opt.value ? "bg-surface-muted font-medium" : "bg-white/70")}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
