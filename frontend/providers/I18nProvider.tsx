"use client";
import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import en from "../i18n/en";
import zhTW from "../i18n/zh-TW";

type Locale = "en" | "zh-TW";
type Dict = Record<string, string>;
type Ctx = {
  locale: Locale;
  t: (key: string, vars?: Record<string, string | number>) => string;
  setLocale: (loc: Locale) => void;
  nfmt: (n: number, opts?: Intl.NumberFormatOptions) => string;
  dfmt: (d: string | number | Date, opts?: Intl.DateTimeFormatOptions) => string;
};

const dicts: Record<Locale, Dict> = { en, "zh-TW": zhTW };

const I18nContext = createContext<Ctx | null>(null);

const normalize = (lng: string | null): Locale => {
  if (!lng) return "en";
  const l = lng.toLowerCase();
  if (l.startsWith("zh") || l.includes("hant")) return "zh-TW";
  return "en";
};

function interpolate(str: string, vars?: Record<string, string | number>) {
  if (!vars) return str;
  return Object.keys(vars).reduce(
    (acc, k) => acc.replace(new RegExp(`{{\\s*${k}\\s*}}`, "g"), String(vars[k])),
    str
  );
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    const stored = typeof window !== "undefined" ? localStorage.getItem("locale") : null;
    const initial = normalize(stored || (typeof navigator !== "undefined" ? navigator.language : "en"));
    setLocaleState(initial);
  }, []);

  useEffect(() => {
    if (typeof document !== "undefined") document.documentElement.lang = locale;
    if (typeof localStorage !== "undefined") localStorage.setItem("locale", locale);
  }, [locale]);

  const setLocale = (loc: Locale) => setLocaleState(loc);

  const t = useMemo(
    () => (key: string, vars?: Record<string, string | number>) => {
      const value = dicts[locale][key] ?? dicts["en"][key] ?? key;
      return interpolate(value, vars);
    },
    [locale]
  );

  const nfmt = (n: number, opts?: Intl.NumberFormatOptions) =>
    new Intl.NumberFormat(locale, opts).format(n);
  const dfmt = (d: string | number | Date, opts?: Intl.DateTimeFormatOptions) =>
    new Intl.DateTimeFormat(locale, opts).format(new Date(d));

  const ctx: Ctx = { locale, t, setLocale, nfmt, dfmt };

  return <I18nContext.Provider value={ctx}>{children}</I18nContext.Provider>;
}

export function useI18nCtx() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("I18nProvider missing");
  return ctx;
}
