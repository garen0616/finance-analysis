import { cn } from "./cn";
export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("glass rounded-md p-4", className)}>{children}</div>;
}
