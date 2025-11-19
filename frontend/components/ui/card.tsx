import { cn } from "./cn";
export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("glass-card", className)}>{children}</div>;
}
