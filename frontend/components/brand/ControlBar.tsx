"use client";
import { useState } from "react";
import SymbolSearch from "../controls/SymbolSearch";
import QuarterPicker from "../controls/QuarterPicker";
import { Button } from "../ui/button";

export default function ControlBar({ symbol, setSymbol, onGenerate }: { symbol: string; setSymbol: (s: string) => void; onGenerate: (mode: "latest" | "specific", year?: number, quarter?: number) => void; }) {
  const [mode, setMode] = useState<"latest" | "specific">("latest");
  const [year, setYear] = useState<number | undefined>(undefined);
  const [quarter, setQuarter] = useState<number | undefined>(undefined);
  return (
    <div className="glass p-4 rounded-md space-y-4">
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1"><SymbolSearch symbol={symbol} setSymbol={setSymbol} /></div>
        <div className="flex-1"><QuarterPicker mode={mode} setMode={setMode} year={year} setYear={setYear} quarter={quarter} setQuarter={setQuarter} /></div>
        <div className="flex items-end">
          <Button
            onClick={() => onGenerate(mode, year, quarter)}
            className="bg-black text-white hover:opacity-90"
            variant="primary"
          >
            Generate
          </Button>
        </div>
      </div>
    </div>
  );
}
