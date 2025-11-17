"use client";
import { useI18n } from "../hooks/useI18n";

export default function QuarterPicker({
  year,
  setYear,
  quarter,
  setQuarter,
}: {
  year?: number;
  setYear: (n?: number) => void;
  quarter?: number;
  setQuarter: (n?: number) => void;
}) {
  const { t } = useI18n();
  return (
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
  );
}
