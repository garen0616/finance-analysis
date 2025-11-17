"use client";
import { cn } from "./cn";
import { forwardRef } from "react";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "outline"; size?: "sm" | "md" };

export const Button = forwardRef<HTMLButtonElement, Props>(function Button({ className, variant = "primary", size = "md", ...props }, ref) {
  const base = "inline-flex items-center justify-center rounded-md font-medium transition-colors";
  const variants = {
    primary: "bg-gradient-to-r from-accent-start to-accent-end text-white shadow-sm hover:opacity-90",
    ghost: "bg-transparent hover:bg-surface-muted text-[var(--fg)]",
    outline: "border border-[var(--line)] bg-white/70 hover:bg-surface-muted",
  } as const;
  const sizes = { sm: "px-3 py-1.5 text-sm", md: "px-4 py-2 text-sm" } as const;
  return <button ref={ref} className={cn(base, variants[variant], sizes[size], className)} {...props} />;
});
