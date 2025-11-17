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
