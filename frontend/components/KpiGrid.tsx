"use client";
import { useI18n } from "../hooks/useI18n";
import { useFormat } from "../lib/format";

export default function KpiGrid({ kpis }: { kpis: Record<string, any> }) {
  const { t } = useI18n();
  const { formatNumber } = useFormat();
  if (!kpis) return null;
  const items: { key: keyof typeof kpis; label: string; style?: string }[] = [
    { key: "revenue", label: t("kpi.revenue") },
    { key: "epsDiluted", label: t("kpi.eps") },
    { key: "grossMarginPct", label: t("kpi.grossMargin") },
    { key: "operatingMarginPct", label: t("kpi.opMargin") },
    { key: "netMarginPct", label: t("kpi.netMargin") },
    { key: "cfo", label: t("kpi.cfo") },
    { key: "capex", label: t("kpi.capex") },
    { key: "fcf", label: t("kpi.fcf") },
    { key: "fcfMarginPct", label: t("kpi.fcfMargin") },
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      {items.map((item) => (
        <div key={item.key} className="p-3 rounded border">
          <div className="text-xs uppercase text-gray-500">{item.label}</div>
          <div className="text-lg font-semibold font-tabular">
            {typeof kpis[item.key] === "number" ? formatNumber(Number(kpis[item.key]), "decimal") : kpis[item.key] ?? "-"}
          </div>
        </div>
      ))}
    </div>
  );
}
