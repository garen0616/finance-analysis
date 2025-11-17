"use client";
import { useEffect, useState } from "react";
import { useI18n } from "../hooks/useI18n";
const backend = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function SymbolSearch({ symbol, setSymbol }: { symbol: string; setSymbol: (s: string) => void }) {
  const { t } = useI18n();
  const [symbols, setSymbols] = useState<any[]>([]);
  const [query, setQuery] = useState("");
  useEffect(() => {
    if (query.length < 1) return;
    const tmr = setTimeout(async () => {
      const res = await fetch(`${backend}/api/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      setSymbols(data || []);
    }, 250);
    return () => clearTimeout(tmr);
  }, [query]);
  return (
    <div>
      <label className="text-sm text-gray-600">{t("controls.search.placeholder")}</label>
      <input
        value={symbol}
        onChange={(e) => {
          setSymbol(e.target.value.toUpperCase());
          setQuery(e.target.value);
        }}
        className="w-full border rounded px-3 py-2"
        list="symbol-options"
      />
      <datalist id="symbol-options">
        {symbols.map((s: any) => (
          <option key={s.symbol} value={s.symbol}>
            {s.name}
          </option>
        ))}
      </datalist>
    </div>
  );
}
