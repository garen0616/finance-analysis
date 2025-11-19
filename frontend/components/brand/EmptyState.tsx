"use client";
export function EmptyState({ message = "Run an analysis to see insights" }: { message?: string }) {
  return (
    <div className="glass-panel rounded-3xl p-6 text-center text-slate-300">
      {message}
    </div>
  );
}
