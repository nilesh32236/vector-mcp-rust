"use client";

import { useEffect } from "react";

interface Shortcut {
  key: string;
  meta?: boolean;
  ctrl?: boolean;
  shift?: boolean;
  handler: () => void;
  preventDefault?: boolean;
  /** Only fire when no input/textarea is focused */
  ignoreInput?: boolean;
}

export function useKeyboard(shortcuts: Shortcut[]) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      for (const s of shortcuts) {
        const metaMatch = s.meta ? (e.metaKey || e.ctrlKey) : true;
        const ctrlMatch = s.ctrl ? e.ctrlKey : true;
        const shiftMatch = s.shift ? e.shiftKey : true;
        const keyMatch = e.key.toLowerCase() === s.key.toLowerCase();

        if (metaMatch && ctrlMatch && shiftMatch && keyMatch) {
          if (s.ignoreInput) {
            const el = document.activeElement;
            if (el?.tagName === "INPUT" || el?.tagName === "TEXTAREA" || el?.getAttribute("contenteditable")) {
              continue;
            }
          }
          if (s.preventDefault !== false) e.preventDefault();
          s.handler();
          return;
        }
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [shortcuts]);
}
