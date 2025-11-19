"use client";
import { Button } from "../ui/button";
export function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="glass-panel rounded-3xl p-5 flex items-center justify-between border border-rose-500/40">
      <div className="text-sm text-rose-200">Something went wrong. Please retry.</div>
      {onRetry && <Button variant="outline" size="sm" onClick={onRetry}>Retry</Button>}
    </div>
  );
}
