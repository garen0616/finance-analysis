"use client";
import { cn } from "../ui/cn";
export default function DataSourceToggle({ value, onChange }: { value: "repo" | "fmp"; onChange: (v: "repo" | "fmp") => void }) {
  const opts = [
    { value: "repo", label: "GitHub Sample" },
    { value: "fmp", label: "FMP Live" },
  ];
  return (
    <div className="flex flex-col gap-2">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Data Source</div>
      <div className="inline-flex rounded-2xl border border-white/15 overflow-hidden bg-white/5">
        {opts.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value as any)}
            className={cn(
              "px-4 py-2 text-sm font-semibold transition-all flex-1",
              value === opt.value
                ? "bg-white/20 text-white shadow-[inset_0_0_12px_rgba(255,255,255,0.35)]"
                : "text-slate-300 hover:text-white"
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
