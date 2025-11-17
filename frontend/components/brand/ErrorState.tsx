"use client";
import { Button } from "../ui/button";
export function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="glass rounded-md p-4 flex items-center justify-between">
      <div className="text-sm text-rose-600">Something went wrong. Please retry.</div>
      {onRetry && <Button variant="outline" size="sm" onClick={onRetry}>Retry</Button>}
    </div>
  );
}
