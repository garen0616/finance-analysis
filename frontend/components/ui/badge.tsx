import { cn } from "./cn";
export function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "positive" | "negative" }) {
  const styles = {
    neutral: "bg-slate-100 text-slate-700",
    positive: "bg-emerald-100 text-emerald-700",
    negative: "bg-rose-100 text-rose-700",
  }[tone];
  return <span className={cn("px-2 py-1 rounded-full text-xs font-medium", styles)}>{children}</span>;
}
