"use client";
import { useI18n } from "../hooks/useI18n";

export default function PeersBlock({ peers }: { peers: any }) {
  const { t } = useI18n();
  if (!peers) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <div className="card p-4 space-y-2">
      <h3 className="text-lg font-semibold">{t("peers.title")}</h3>
      <div className="text-sm text-gray-600">{t("peers.median")}: {JSON.stringify(peers.medians)}</div>
      <div className="text-sm text-gray-600">{t("peers.vsCompany")}: {peers.peers?.join(", ")}</div>
    </div>
  );
}
