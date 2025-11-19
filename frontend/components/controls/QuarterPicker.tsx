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
    <div className="space-y-3">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Mode</div>
      <div className="inline-flex rounded-2xl border border-white/15 overflow-hidden bg-white/5">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => setMode(opt)}
            className={cn(
              "px-4 py-2 text-sm font-semibold transition-all",
              mode === opt ? "bg-white/15 text-white" : "text-slate-400 hover:text-white"
            )}
          >
            {opt === "latest" ? "Latest" : "Specific"}
          </button>
        ))}
      </div>
      {mode === "specific" && (
        <div className="flex gap-3">
          <input
            type="number"
            placeholder="Year"
            value={year ?? ""}
            onChange={(e) => setYear(e.target.value ? Number(e.target.value) : undefined)}
            className="glass-input flex-1"
          />
          <select
            value={quarter ?? ""}
            onChange={(e) => setQuarter(e.target.value ? Number(e.target.value) : undefined)}
            className="glass-input flex-1"
          >
            <option value="">Quarter</option>
            {[1, 2, 3, 4].map((q) => (
              <option key={q} value={q} className="bg-slate-900">{`Q${q}`}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
