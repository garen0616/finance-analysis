"use client";
import { useState } from "react";
import { cn } from "./cn";
export function Tooltip({ label, children }: { label: string; children: React.ReactNode }) {
  const [hover, setHover] = useState(false);
  return (
    <span className="relative inline-flex"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
      aria-label={label}
    >
      {children}
      {hover && (
        <span className={cn("absolute z-50 whitespace-nowrap rounded-md bg-slate-900 text-white text-xs px-2 py-1 shadow", "translate-y-[-8px] left-1/2 -translate-x-1/2")}>{label}</span>
      )}
    </span>
  );
}
