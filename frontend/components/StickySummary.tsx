"use client";
import { useI18n } from "../hooks/useI18n";

export default function StickySummary({ title, bullets }: { title: string; bullets: string[] }) {
  const { t } = useI18n();
  return (
    <div className="card p-4 space-y-2 sticky top-0 z-10">
      <h2 className="text-xl font-semibold">{title || t("summary.title")}</h2>
      <ul className="list-disc pl-5 space-y-1">
        {bullets?.map((b, i) => (
          <li key={i}>{b}</li>
        ))}
      </ul>
    </div>
  );
}
