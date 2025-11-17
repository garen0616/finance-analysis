"use client";
import { useI18n } from "../hooks/useI18n";
import clsx from "clsx";

export default function LanguageToggle() {
  const { locale, setLocale, t } = useI18n();
  const opts: { label: string; value: "en" | "zh-TW" }[] = [
    { label: "EN", value: "en" },
    { label: "繁", value: "zh-TW" },
  ];
  return (
    <div className="inline-flex border rounded-full overflow-hidden">
      {opts.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={locale === o.value}
          aria-label={t("app.title")}
          onClick={() => setLocale(o.value)}
          className={clsx(
            "px-3 py-1 text-sm",
            locale === o.value ? "bg-indigo-600 text-white" : "bg-transparent text-gray-700 dark:text-gray-200"
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
