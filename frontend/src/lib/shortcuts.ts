/**
 * Global keyboard shortcut definitions.
 *
 * Each entry describes a shortcut with:
 * - `key`   — the KeyboardEvent.key value
 * - `meta`  — whether Cmd/Ctrl must be held
 * - `label` — human-readable display string
 */
export const KEYBOARD_SHORTCUTS = {
  /** Focus the main search input */
  FOCUS_SEARCH: {
    key: "k",
    meta: true,
    label: "⌘K / Ctrl+K",
  },
  /** Focus the main search input via forward-slash (when not in an input) */
  FOCUS_SEARCH_SLASH: {
    key: "/",
    meta: false,
    label: "/",
  },
  /** Clear query / close modal / hide suggestions */
  CLEAR_SEARCH: {
    key: "Escape",
    meta: false,
    label: "Esc",
  },
  /** Submit the current query */
  SUBMIT_SEARCH: {
    key: "Enter",
    meta: true,
    label: "⌘Enter / Ctrl+Enter",
  },
} as const;

export type ShortcutKey = keyof typeof KEYBOARD_SHORTCUTS;
