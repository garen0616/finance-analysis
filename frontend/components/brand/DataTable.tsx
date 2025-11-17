"use client";
import { Button } from "../ui/button";

export default function DataTable({ title, rows }: { title: string; rows: any[] }) {
  return (
    <div className="glass rounded-md p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold">{title}</div>
        <Button variant="outline" size="sm">CSV</Button>
      </div>
      <div className="overflow-auto max-h-96">
        <table className="min-w-full text-sm">
          <thead className="sticky top-0 bg-white/90 backdrop-blur">
            <tr>
              {rows?.[0] ? Object.keys(rows[0]).map((k) => <th key={k} className="text-left px-3 py-2 text-slate-500">{k}</th>) : <th>—</th>}
            </tr>
          </thead>
          <tbody>
            {rows?.length ? rows.map((r, i) => (
              <tr key={i} className="border-t border-[var(--line)]">
                {Object.values(r).map((v, j) => (
                  <td key={j} className="px-3 py-2 whitespace-nowrap">{typeof v === "number" ? v.toLocaleString() : String(v ?? "")}</td>
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
