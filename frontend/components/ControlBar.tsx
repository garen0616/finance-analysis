"use client";
import { useI18n } from "../hooks/useI18n";

type Props = {
  symbol: string;
  setSymbol: (s: string) => void;
  year?: number;
  setYear: (n?: number) => void;
  quarter?: number;
  setQuarter: (n?: number) => void;
  mode: "latest" | "specific";
  setMode: (m: "latest" | "specific") => void;
  onGenerate: () => void;
};

export default function ControlBar(props: Props) {
  const { t } = useI18n();
  const { symbol, setSymbol, year, setYear, quarter, setQuarter, mode, setMode, onGenerate } = props;
  return (
    <div className="card p-4 space-y-3">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div>
          <label className="text-sm text-gray-600">{t("controls.search.placeholder")}</label>
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="text-sm text-gray-600">{t("controls.specific")}</label>
          <select value={mode} onChange={(e) => setMode(e.target.value as any)} className="w-full border rounded px-3 py-2">
            <option value="latest">{t("controls.latest")}</option>
            <option value="specific">{t("controls.specific")}</option>
          </select>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-sm text-gray-600">{t("controls.year")}</label>
            <input
              type="number"
              value={year ?? ""}
              onChange={(e) => setYear(e.target.value ? Number(e.target.value) : undefined)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="text-sm text-gray-600">{t("controls.quarter")}</label>
            <input
              type="number"
              min={1}
              max={4}
              value={quarter ?? ""}
              onChange={(e) => setQuarter(e.target.value ? Number(e.target.value) : undefined)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
        </div>
      </div>
      <button onClick={onGenerate} className="px-4 py-2 bg-blue-600 text-white rounded">
        {t("actions.generate")}
      </button>
    </div>
  );
}
