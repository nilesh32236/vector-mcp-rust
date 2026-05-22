export interface McpResponse {
  result?: string;
  error?: string;
}

export async function mcpCall(toolName: string, args: Record<string, unknown>): Promise<string> {
  const res = await fetch("/api/mcp/call", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool: toolName, arguments: args }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data: McpResponse = await res.json();
  if (data.error) throw new Error(data.error);
  return data.result ?? "";
}

export interface HistoryEntry {
  id: string;
  tool: string;
  action: string;
  label: string;
  args: Record<string, unknown>;
  result: string;
  timestamp: number;
}

const HISTORY_KEY = "mcp_history";

export function getHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function addHistory(entry: Omit<HistoryEntry, "id" | "timestamp">): void {
  try {
    const history = getHistory();
    const newEntry: HistoryEntry = {
      ...entry,
      id: crypto.randomUUID(),
      timestamp: Date.now(),
    };
    history.unshift(newEntry);
    if (history.length > 100) history.length = 100;
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // storage full or unavailable
  }
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {
    // silent
  }
}
