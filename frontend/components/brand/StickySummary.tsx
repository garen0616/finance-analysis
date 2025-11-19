"use client";
import { Badge } from "../ui/badge";

export default function StickySummary({ bullets }: { bullets: string[] }) {
  if (!bullets?.length) return null;
  return (
    <div className="glass-panel rounded-3xl p-5 sticky top-6 z-30 border border-white/15 shadow-glow">
      <div className="text-xs uppercase tracking-[0.4em] text-slate-400 mb-3">Executive Summary</div>
      <div className="flex flex-wrap gap-2">
        {bullets.slice(0, 6).map((b, i) => (
          <Badge key={i}>{b}</Badge>
        ))}
      </div>
    </div>
  );
}
