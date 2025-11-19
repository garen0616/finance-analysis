import { cn } from "./cn";

export function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "positive" | "negative" }) {
  const toneStyles = {
    neutral: "bg-white/10 text-slate-200 border border-white/15",
    positive: "bg-emerald-400/20 text-emerald-100 border border-emerald-300/40",
    negative: "bg-rose-500/20 text-rose-100 border border-rose-400/40",
  }[tone];
  return (
    <span className={cn("px-3 py-1 rounded-full text-xs font-semibold tracking-wide backdrop-blur", toneStyles)}>
      {children}
    </span>
  );
}
