"use client";
import { useI18n } from "../hooks/useI18n";

export default function DataTables({ tables }: { tables: any }) {
  const { t } = useI18n();
  if (!tables) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <div className="card p-4">
      <h3 className="text-lg font-semibold mb-2">{t("data.title")}</h3>
      <pre className="text-xs overflow-auto max-h-96">{JSON.stringify(tables, null, 2)}</pre>
    </div>
  );
}
