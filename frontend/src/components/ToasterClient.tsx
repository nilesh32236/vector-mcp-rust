"use client";

import { useTheme } from "next-themes";
import { Toaster } from "sonner";

/**
 * Thin client wrapper so the Toaster can read the resolved theme from
 * next-themes and stay in sync with the user's dark/light/system preference.
 *
 * This component must be a Client Component because `useTheme` relies on
 * React context, which is unavailable in Server Components.
 */
export function ToasterClient() {
  const { resolvedTheme } = useTheme();
  return (
    <Toaster
      position="bottom-right"
      theme={(resolvedTheme as "dark" | "light" | "system") ?? "dark"}
      richColors
      closeButton
      duration={4000}
    />
  );
}
