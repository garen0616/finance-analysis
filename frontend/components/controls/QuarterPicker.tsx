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
