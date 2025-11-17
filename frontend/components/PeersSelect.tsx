"use client";
import { useI18n } from "../hooks/useI18n";

export default function PeersSelect({ peers, setPeers }: { peers: string[]; setPeers: (p: string[]) => void }) {
  const { t } = useI18n();
  const addPeer = () => setPeers([...(peers || []), ""]);
  return (
    <div>
      <label className="text-sm text-gray-600">{t("peers.title")}</label>
      <div className="space-y-2">
        {(peers || []).map((p, i) => (
          <input
            key={i}
            value={p}
            onChange={(e) => {
              const clone = [...peers];
              clone[i] = e.target.value.toUpperCase();
              setPeers(clone);
            }}
            className="w-full border rounded px-3 py-2"
          />
        ))}
        <button type="button" onClick={addPeer} className="text-sm text-blue-600 underline">
          {t("actions.addPeers")}
        </button>
      </div>
    </div>
  );
}
