"use client";
import dynamic from "next/dynamic";
import { useState } from "react";
import { Button } from "../ui/button";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function GraphPreview({ enabled, data }: { enabled?: boolean; data: { nodes: any[]; links: any[] } }) {
  const [open, setOpen] = useState(true);
  if (!enabled) return null;
  return (
    <div className="glass-panel rounded-3xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold text-white">Knowledge Graph Preview</div>
        <Button variant="ghost" size="sm" onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"}</Button>
      </div>
      {open && (
        <div className="h-80 rounded-2xl border border-white/10 overflow-hidden">
          <ForceGraph2D graphData={data} nodeLabel={(n: any) => n.id} nodeAutoColorBy="label" />
        </div>
      )}
    </div>
  );
}
