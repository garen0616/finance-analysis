"use client";
import { useI18n } from "../hooks/useI18n";
import clsx from "clsx";

export default function SectionTabs({ active, setActive }: { active: string; setActive: (k: string) => void }) {
  const { t } = useI18n();
  const tabs = [
    { key: "overview", label: t("nav.overview") },
    { key: "financials", label: t("nav.financials") },
    { key: "transcript", label: t("nav.transcript") },
    { key: "peers", label: t("nav.peers") },
    { key: "graph", label: t("nav.graph") },
    { key: "data", label: t("nav.data") },
  ];
  return (
    <div role="tablist" className="flex gap-2 border-b overflow-x-auto">
      {tabs.map((tab) => (
        <button
          role="tab"
          key={tab.key}
          aria-pressed={active === tab.key}
          onClick={() => setActive(tab.key)}
          className={clsx("px-3 py-2 text-sm", active === tab.key ? "border-b-2 border-blue-600 font-semibold" : "text-gray-500")}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
