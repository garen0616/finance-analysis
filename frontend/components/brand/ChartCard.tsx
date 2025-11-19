"use client";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function ChartCard({ title, desc, dataUrl }: { title: string; desc?: string; dataUrl?: string }) {
  return (
    <div className="glass-panel rounded-3xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold text-white">{title}</div>
          {desc && <div className="text-xs text-slate-400">{desc}</div>}
        </div>
      </div>
      {dataUrl ? (
        <img src={dataUrl} alt={title} className="w-full h-auto rounded-2xl border border-white/10" />
      ) : (
        <div className="text-sm text-slate-400">No chart data</div>
      )}
    </div>
  );
}
