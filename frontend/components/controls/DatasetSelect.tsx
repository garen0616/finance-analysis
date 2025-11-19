"use client";
import useSWR from "swr";
import { apiBase } from "../../lib/api";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function DatasetSelect({ value, onChange }: { value?: string; onChange: (v?: string) => void }) {
  const { data, isLoading } = useSWR(`${apiBase}/api/datasources`, fetcher, { dedupingInterval: 60000 });
  const list = data?.datasets || [];
  return (
    <div className="space-y-2">
      <label className="text-xs uppercase tracking-[0.3em] text-slate-400">Dataset</label>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value || undefined)}
        className="w-full glass-input bg-white/5"
      >
        <option value="">{isLoading ? "Loading..." : "Select dataset"}</option>
        {list.map((d: any) => (
          <option key={d.name} value={d.name} className="bg-night-900">{d.name}</option>
        ))}
      </select>
    </div>
  );
}
