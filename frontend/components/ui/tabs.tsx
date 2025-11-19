"use client";
import { useId } from "react";
import { cn } from "./cn";

type Tab = { key: string; label: string };

export function Tabs({ tabs, active, onChange }: { tabs: Tab[]; active: string; onChange: (k: string) => void }) {
  const id = useId();
  return (
    <div role="tablist" aria-label="sections" className="flex gap-2 overflow-x-auto">
      {tabs.map((t) => (
        <button
          key={t.key}
          id={`${id}-${t.key}`}
          role="tab"
          aria-selected={active === t.key}
          onClick={() => onChange(t.key)}
          className={cn(
            "px-4 py-2 rounded-2xl text-sm font-medium transition-all border backdrop-blur",
            active === t.key
              ? "bg-white/15 border-white/30 text-white shadow-glow"
              : "bg-white/5 border-white/10 text-slate-300 hover:text-white"
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
