"use client";
import { Badge } from "../ui/badge";

export default function StickySummary({ bullets }: { bullets: string[] }) {
  if (!bullets?.length) return null;
  return (
    <div className="sticky top-0 z-50 glass rounded-md p-4 border-b-2 border-gradient-to-r from-accent-500 to-accent-end shadow-md backdrop-blur">
      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Executive Summary</div>
      <div className="flex flex-wrap gap-2">
        {bullets.slice(0, 6).map((b, i) => (
          <Badge key={i} tone="neutral">{b}</Badge>
        ))}
      </div>
    </div>
  );
}
