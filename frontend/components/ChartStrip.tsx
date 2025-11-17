"use client";
import { useI18n } from "../hooks/useI18n";

export default function ChartStrip({ charts }: { charts: Record<string, string> }) {
  const { t } = useI18n();
  if (!charts) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {Object.entries(charts).map(([k, v]) => (
        <div key={k} className="card p-2">
          <div className="text-sm font-semibold px-2 py-1">{k}</div>
          <img src={v} alt={k} className="w-full" />
        </div>
      ))}
    </div>
  );
}
