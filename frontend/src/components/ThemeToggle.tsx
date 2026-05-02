"use client";

import { useTheme } from "next-themes";
import { Sun, Moon, Monitor } from "lucide-react";
import { useEffect, useState } from "react";

const TOGGLE_CLASS =
  "p-2 rounded-lg hover:bg-white/5 transition-colors cursor-pointer text-foreground/40 hover:text-foreground/70";

/** Map theme name → icon component. Falls back to Monitor for unknown values. */
function ThemeIcon({ theme }: { theme: string | undefined }) {
  if (theme === "dark") return <Moon className="w-4 h-4" />;
  if (theme === "light") return <Sun className="w-4 h-4" />;
  return <Monitor className="w-4 h-4" />;
}

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch — only render the real toggle after mount.
  useEffect(() => setMounted(true), []);

  // Render a same-size placeholder before mount to prevent layout shift.
  if (!mounted) {
    return (
      <span
        className={TOGGLE_CLASS}
        aria-hidden
        role="presentation"
        style={{ display: "inline-flex", width: "2rem", height: "2rem" }}
      />
    );
  }

  const cycle = () => {
    if (theme === "dark") setTheme("light");
    else if (theme === "light") setTheme("system");
    else setTheme("dark");
  };

  const label =
    theme === "dark"
      ? "Switch to light mode"
      : theme === "light"
      ? "Switch to system mode"
      : "Switch to dark mode";

  return (
    <button
      onClick={cycle}
      aria-label={label}
      title={label}
      className={TOGGLE_CLASS}
    >
      <ThemeIcon theme={theme} />
    </button>
  );
}
