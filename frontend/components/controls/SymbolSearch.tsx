"use client";
import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { apiBase } from "../../lib/api";
import { cn } from "../ui/cn";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function SymbolSearch({ symbol, setSymbol }: { symbol: string; setSymbol: (s: string) => void }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const { data } = useSWR(query.length > 0 ? `${apiBase}/api/search?q=${encodeURIComponent(query)}` : null, fetcher, { dedupingInterval: 400 });
  const list = useMemo(() => data || [], [data]);
  return (
    <div className="relative">
      <label className="text-xs uppercase tracking-[0.3em] text-slate-400">Symbol</label>
      <input
        value={symbol}
        onChange={(e) => { setSymbol(e.target.value.toUpperCase()); setQuery(e.target.value); setOpen(true); }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        className="w-full mt-2 glass-input placeholder:text-slate-500"
        placeholder="Search or type symbol"
      />
      {open && list.length > 0 && (
        <div className="absolute mt-2 w-full glass-panel rounded-2xl max-h-64 overflow-auto z-40 p-2 space-y-1">
          {list.map((item: any) => (
            <div
              key={item.symbol}
              className={cn("px-3 py-2 cursor-pointer rounded-xl hover:bg-white/10 transition-colors")}
              onMouseDown={() => { setSymbol(item.symbol); setOpen(false); }}
            >
              <div className="font-semibold text-white">{item.symbol}</div>
              <div className="text-xs text-slate-400">{item.name}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
