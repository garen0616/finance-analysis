"use client";
import dynamic from "next/dynamic";
import { useState } from "react";
import { Button } from "../ui/button";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function GraphPreview({ enabled, data }: { enabled?: boolean; data: { nodes: any[]; links: any[] } }) {
  const [open, setOpen] = useState(true);
  if (!enabled) return null;
  return (
    <div className="glass rounded-md p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold">Graph Preview</div>
        <Button variant="ghost" size="sm" onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"}</Button>
      </div>
      {open && (
        <div className="h-80">
          <ForceGraph2D graphData={data} nodeLabel={(n: any) => n.id} nodeAutoColorBy="label" />
        </div>
      )}
    </div>
  );
}
