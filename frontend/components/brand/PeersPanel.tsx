"use client";
import { useState } from "react";
import { Badge } from "../ui/badge";

export default function PeersPanel({ peers }: { peers: { peers?: any[]; medians?: any } }) {
  const list = peers?.peers || [];
  const [selected, setSelected] = useState<string | null>(null);
  return (
    <div className="glass-panel rounded-3xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold text-white">Peer Comparison</div>
        <span className="text-xs uppercase tracking-[0.3em] text-slate-400">Sector pulse</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {list.length ? (
          list.map((p: any) => {
            const sym = p.symbol || p;
            return (
              <button key={sym} onClick={() => setSelected(sym)} type="button">
                <Badge tone={selected === sym ? "positive" : "neutral"}>{sym}</Badge>
              </button>
            );
          })
        ) : (
          <p className="text-sm text-slate-400">No peers available</p>
        )}
      </div>
      {peers?.medians && (
        <div className="text-sm text-slate-300 font-mono bg-white/5 rounded-2xl p-3 border border-white/10">
          {Object.entries(peers.medians as Record<string, unknown>).map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="text-slate-400">{k}</span>
              <span className="text-white">
                {typeof v === "number" ? v.toFixed(2) : String(v ?? "—")}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
