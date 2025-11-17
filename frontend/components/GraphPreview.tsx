"use client";
import dynamic from "next/dynamic";
import { useI18n } from "../hooks/useI18n";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function GraphPreview({ enabled, data }: { enabled: boolean; data: { nodes: any[]; links: any[] } }) {
  const { t } = useI18n();
  if (!enabled) return <div className="text-sm text-gray-500">{t("empty.noData")}</div>;
  return (
    <div className="card p-4">
      <h3 className="text-lg font-semibold mb-2">{t("graph.title")}</h3>
      <div className="h-80">
        <ForceGraph2D graphData={data} nodeLabel={(n: any) => n.id} nodeAutoColorBy="label" />
      </div>
    </div>
  );
}
