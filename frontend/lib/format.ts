import { useI18n } from "../hooks/useI18n";

export function useFormat() {
  const { locale } = useI18n();
  const formatNumber = (n: number, style: "decimal" | "currency" | "percent" = "decimal", currency = "USD") =>
    new Intl.NumberFormat(locale, { style, currency, maximumFractionDigits: style === "percent" ? 2 : 2 }).format(n);
  const formatSignedPercent = (x: number) => {
    const sign = x > 0 ? "+" : "";
    return `${sign}${formatNumber(x / 100, "percent")}`;
  };
  const formatDate = (iso: string) => new Intl.DateFormat(locale, { year: "numeric", month: "2-digit", day: "2-digit" } as any).format(new Date(iso));
  return { formatNumber, formatSignedPercent, formatDate };
}
