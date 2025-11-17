"use client";
import LanguageToggle from "./LanguageToggle";
import { useI18n } from "../hooks/useI18n";

export default function BrandHeader() {
  const { t } = useI18n();
  return (
    <header className="flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-semibold">{t("app.title")}</h1>
        <p className="text-sm text-gray-500">FMP + RAG + PDF</p>
      </div>
      <div className="flex items-center gap-3">
        <LanguageToggle />
      </div>
    </header>
  );
}
