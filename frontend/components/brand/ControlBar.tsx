"use client";
import { useEffect } from "react";
import SymbolSearch from "../controls/SymbolSearch";
import QuarterPicker from "../controls/QuarterPicker";
import DataSourceToggle from "../controls/DataSourceToggle";
import DatasetSelect from "../controls/DatasetSelect";
import TickerSelect from "../controls/TickerSelect";
import QuarterSelectRepo from "../controls/QuarterSelectRepo";
import { Button } from "../ui/button";

type Props = {
  source: "repo" | "fmp";
  setSource: (s: "repo" | "fmp") => void;
  dataset?: string;
  setDataset: (s?: string) => void;
  symbol: string;
  setSymbol: (s: string) => void;
  mode: "latest" | "specific";
  setMode: (m: "latest" | "specific") => void;
  year?: number;
  setYear: (n?: number) => void;
  quarter?: number;
  setQuarter: (n?: number) => void;
  onGenerate: () => void;
};

export default function ControlBar({
  source,
  setSource,
  dataset,
  setDataset,
  symbol,
  setSymbol,
  mode,
  setMode,
  year,
  setYear,
  quarter,
  setQuarter,
  onGenerate,
}: Props) {
  useEffect(() => {
    if (source === "repo") {
      setMode("specific");
    }
  }, [source, setMode]);

  return (
    <div className="glass p-4 rounded-md space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <DataSourceToggle value={source} onChange={(v) => { setSource(v); if (v === "repo") setMode("specific"); }} />
        {source === "repo" ? (
          <>
            <DatasetSelect value={dataset} onChange={setDataset} />
            <TickerSelect dataset={dataset} value={symbol} onChange={setSymbol} />
            <QuarterSelectRepo dataset={dataset} symbol={symbol} year={year} quarter={quarter} onChange={(y, q) => { setYear(y); setQuarter(q); }} />
          </>
        ) : (
          <>
            <div className="md:col-span-2">
              <SymbolSearch symbol={symbol} setSymbol={setSymbol} />
            </div>
            <QuarterPicker mode={mode} setMode={setMode} year={year} setYear={setYear} quarter={quarter} setQuarter={setQuarter} />
          </>
        )}
      </div>
      <div className="flex justify-end">
        <Button
          onClick={onGenerate}
          className="bg-black text-white hover:opacity-90"
          variant="primary"
        >
          Generate
        </Button>
      </div>
    </div>
  );
}
