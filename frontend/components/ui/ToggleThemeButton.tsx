"use client";
import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "./button";

export function ToggleThemeButton() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    setDark(media.matches);
  }, []);
  useEffect(() => { document.documentElement.classList.toggle("dark", dark); }, [dark]);
  return (
    <Button variant="ghost" size="sm" onClick={() => setDark((d) => !d)} aria-label="Toggle theme">
      {dark ? <Sun size={16} /> : <Moon size={16} />}
    </Button>
  );
}
