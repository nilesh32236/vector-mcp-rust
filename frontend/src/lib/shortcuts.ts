export const KEYBOARD_SHORTCUTS = {
  FOCUS_SEARCH:        { key: "k",   meta: true,  label: "⌘K / Ctrl+K" },
  FOCUS_SEARCH_SLASH:  { key: "/",   meta: false, label: "/" },
  CLEAR_SEARCH:        { key: "Escape", meta: false, label: "Esc" },
  SUBMIT_SEARCH:       { key: "Enter", meta: true,  label: "⌘Enter / Ctrl+Enter" },
} as const;

export const PANEL_SHORTCUTS = [
  { key: "1", label: "Search", hint: "⌘1" },
  { key: "2", label: "Analyze", hint: "⌘2" },
  { key: "3", label: "Modify", hint: "⌘3" },
  { key: "4", label: "Workspace", hint: "⌘4" },
  { key: "5", label: "LSP", hint: "⌘5" },
] as const;
