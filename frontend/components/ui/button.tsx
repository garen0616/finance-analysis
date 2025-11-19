"use client";
import { cn } from "./cn";
import { forwardRef } from "react";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "outline"; size?: "sm" | "md" };

export const Button = forwardRef<HTMLButtonElement, Props>(function Button({ className, variant = "primary", size = "md", ...props }, ref) {
  const base = "inline-flex items-center justify-center rounded-2xl font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-transparent focus-visible:ring-cyan-400";
  const variants = {
    primary: "bg-gradient-to-r from-cyan-400/80 via-sky-500/80 to-blue-500/80 text-white shadow-glow hover:shadow-[0_0_30px_rgba(56,189,248,0.45)]",
    ghost: "bg-white/5 text-slate-100 border border-white/10 hover:bg-white/10",
    outline: "bg-transparent border border-white/20 text-white hover:border-cyan-300/50",
  } as const;
  const sizes = { sm: "px-4 py-2 text-xs tracking-wide", md: "px-5 py-2.5 text-sm" } as const;
  return <button ref={ref} className={cn(base, variants[variant], sizes[size], className)} {...props} />;
});
