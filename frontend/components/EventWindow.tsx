"use client";
import { useI18n } from "../hooks/useI18n";
import { useFormat } from "../lib/format";

export default function EventWindow({ events }: { events: any[] }) {
  const { t } = useI18n();
  const { formatSignedPercent } = useFormat();
  if (!events || !events.length) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left">
          <th className="py-1">{t("event.title")}</th>
          <th className="py-1">{t("event.shock")}</th>
        </tr>
      </thead>
      <tbody>
        {events.map((e, i) => (
          <tr key={i} className="border-t">
            <td className="py-1">{e.window}</td>
            <td className="py-1">
              {formatSignedPercent(e.returnPct || 0)} {e.shock ? `(${t("event.shock")})` : `(${t("event.noShock")})`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
