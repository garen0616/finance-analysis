"use client";
import { Button } from "../ui/button";
import { ToggleThemeButton } from "../ui/ToggleThemeButton";
import { Download, FileDown } from "lucide-react";
import { pdfUrl } from "../../lib/api";

export default function Header({ analysisId }: { analysisId?: string }) {
  return (
    <div className="flex items-center justify-between py-4">
      <div className="flex items-center gap-3">
        <img src="/brand.svg" alt="brand" className="h-8 w-8" />
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500">Finance Analytics</div>
          <div className="text-xl font-semibold">Earnings Insight</div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <ToggleThemeButton />
        <Button variant="outline" size="sm" disabled={!analysisId} onClick={() => analysisId && window.open(pdfUrl(analysisId), "_blank")}> <Download className="mr-2 h-4 w-4" /> PDF </Button>
        <Button variant="ghost" size="sm"> <FileDown className="mr-2 h-4 w-4" /> CSV </Button>
      </div>
    </div>
  );
}
