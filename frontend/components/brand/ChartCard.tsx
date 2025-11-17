"use client";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function ChartCard({ title, desc, dataUrl }: { title: string; desc?: string; dataUrl?: string }) {
  return (
    <div className="glass rounded-md p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold">{title}</div>
          {desc && <div className="text-xs text-slate-500">{desc}</div>}
        </div>
      </div>
      {dataUrl ? (
        <img src={dataUrl} alt={title} className="w-full h-auto rounded" />
      ) : (
        <div className="text-sm text-slate-500">No chart data</div>
      )}
    </div>
  );
}
