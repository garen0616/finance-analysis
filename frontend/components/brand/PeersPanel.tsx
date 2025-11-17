"use client";
import { useState } from "react";
import { Badge } from "../ui/badge";

export default function PeersPanel({ peers }: { peers: { peers?: any[]; medians?: any } }) {
  const list = peers?.peers || [];
  const [selected, setSelected] = useState<string | null>(null);
  return (
    <div className="glass rounded-md p-4 space-y-3">
      <div className="text-lg font-semibold">Peers</div>
      <div className="flex flex-wrap gap-2">
        {list.map((p: any) => (
          <Badge key={p.symbol || p} tone={selected === (p.symbol || p) ? "positive" : "neutral"}>{p.symbol || p}</Badge>
        ))}
      </div>
      <div className="text-sm text-slate-500">Medians: {JSON.stringify(peers?.medians || {})}</div>
    </div>
  );
}
