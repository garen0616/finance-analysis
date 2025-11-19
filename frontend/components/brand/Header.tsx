"use client";
import { Button } from "../ui/button";
import { Download, FileDown } from "lucide-react";
import { pdfUrl } from "../../lib/api";

export default function Header({ analysisId }: { analysisId?: string }) {
  return (
    <div className="glass-panel px-6 py-5 flex items-center justify-between rounded-3xl">
      <div>
        <p className="text-xs uppercase tracking-[0.4em] text-slate-400">Earnings Intelligence</p>
        <p className="text-3xl font-semibold text-white mt-1">Prime Analytics Studio</p>
        <p className="text-sm text-slate-400 mt-1">Blend agentic RAG, transcripts, and peer insights across live + sample data sources.</p>
      </div>
      <div className="flex items-center gap-3">
        <Button variant="outline" size="sm" disabled={!analysisId} onClick={() => analysisId && window.open(pdfUrl(analysisId), "_blank")}>
          <Download className="mr-2 h-4 w-4" /> Download PDF
        </Button>
        <Button variant="ghost" size="sm" disabled={!analysisId}>
          <FileDown className="mr-2 h-4 w-4" /> Export CSV
        </Button>
      </div>
    </div>
  );
}
