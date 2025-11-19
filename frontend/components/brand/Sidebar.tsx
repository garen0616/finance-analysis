"use client";
import { AreaChart, BarChart3, Database, LineChart, Network, Settings } from "lucide-react";
import { cn } from "../ui/cn";

const items = [
  { key: "overview", label: "Overview", icon: AreaChart },
  { key: "financials", label: "Financials", icon: LineChart },
  { key: "transcript", label: "Transcript", icon: BarChart3 },
  { key: "graph", label: "Graph", icon: Network },
  { key: "data", label: "Data Room", icon: Database },
];

export function Sidebar({ active, onSelect }: { active: string; onSelect?: (key: string) => void }) {
  return (
    <aside className="glass-panel w-64 shrink-0 p-6 flex flex-col gap-8 text-slate-200">
      <div className="flex items-center gap-3">
        <div className="h-12 w-12 rounded-2xl bg-white/10 flex items-center justify-center text-xl font-bold text-cyan-300 shadow-glow">
          FA
        </div>
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Prime</p>
          <p className="text-xl font-semibold text-white">Earnings Hub</p>
        </div>
      </div>
      <nav className="flex flex-col gap-2 text-sm">
        {items.map((item) => {
          const Icon = item.icon;
          const isActive = active === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onSelect?.(item.key)}
              className={cn(
                "flex items-center gap-3 w-full px-4 py-2 rounded-2xl text-left transition-all",
                "border border-white/10 hover:border-cyan-300/30",
                isActive
                  ? "text-[#FFD700] border-[#FFD700]/60 bg-white/10 shadow-[0_0_25px_rgba(250,204,21,0.4)]"
                  : "text-slate-300"
              )}
            >
              <Icon size={18} className={cn(isActive ? "text-[#FFD700]" : "text-white")} />
              <span className="font-semibold">{item.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="mt-auto glass-panel p-4 rounded-3xl border border-white/15 bg-white/5">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400 mb-2">Status</p>
        <div className="flex items-center justify-between text-sm">
          <span>Realtime</span>
          <span className="text-emerald-300 font-semibold">Online</span>
        </div>
        <div className="flex items-center justify-between text-sm mt-1">
          <span>Source</span>
          <span className="text-sky-300">FMP / Repo</span>
        </div>
        <button className="mt-4 w-full inline-flex items-center justify-center gap-2 text-xs text-slate-100 bg-white/10 border border-white/20 rounded-2xl py-2">
          <Settings size={14} /> Preferences
        </button>
      </div>
    </aside>
  );
}
