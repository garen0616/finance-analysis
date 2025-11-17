export function EmptyState({ message = "No data available" }: { message?: string }) {
  return <div className="text-sm text-slate-500">{message}</div>;
}
