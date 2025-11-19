"use client";
import { Button } from "../ui/button";
import { Download } from "lucide-react";

export default function DataTable({ title, rows }: { title: string; rows: any[] }) {
  return (
    <div className="glass-panel rounded-3xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-white">{title}</p>
          <p className="text-xs text-slate-400">Raw quarterly tables and YoY/QoQ growth</p>
        </div>
        <Button variant="ghost" size="sm">
          <Download className="mr-2 h-4 w-4" /> CSV
        </Button>
      </div>
      <div className="overflow-auto max-h-96">
        <table className="min-w-full text-sm text-slate-200">
          <thead className="sticky top-0 bg-night-900/70 backdrop-blur">
            <tr>
              {rows?.[0] ? Object.keys(rows[0]).map((k) => (
                <th key={k} className="text-left px-3 py-2 text-xs text-slate-400 uppercase">{k}</th>
              )) : <th className="text-left px-3 py-2 text-xs text-slate-400 uppercase">metric</th>}
            </tr>
          </thead>
          <tbody>
            {rows?.length ? rows.map((r, i) => (
              <tr key={i} className="border-t border-white/5">
                {Object.values(r).map((v, j) => (
                  <td key={j} className="px-3 py-2 whitespace-nowrap">
                    {typeof v === "number" ? v.toLocaleString() : String(v ?? "")}
                  </td>
                ))}
              </tr>
            )) : (
              <tr><td className="px-3 py-4 text-slate-500">No data</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
