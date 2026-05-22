"use client";

import { useState } from "react";
import { Settings, Play, Loader2, AlertCircle } from "lucide-react";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { mcpCall, addHistory } from "@/lib/api";
import { ResultCopyButton } from "./ResultCopyButton";

type WorkspaceAction = "store_context" | "delete_context" | "get_indexing_diagnostics" | "trigger_index";

const actions: { value: WorkspaceAction; label: string }[] = [
  { value: "store_context", label: "Store Context" },
  { value: "delete_context", label: "Delete Context" },
  { value: "get_indexing_diagnostics", label: "Get Diagnostics" },
  { value: "trigger_index", label: "Trigger Re-index" },
];

interface WorkspacePanelProps {
  onResult?: () => void;
}

export function WorkspacePanel({ onResult }: WorkspacePanelProps) {
  const [action, setAction] = useState<WorkspaceAction>("store_context");
  const [text, setText] = useState("");
  const [targetPath, setTargetPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const args: Record<string, unknown> = { action };

      switch (action) {
        case "store_context":
          args.text = text;
          args.target_path = targetPath;
          break;
        case "delete_context":
          args.target_path = targetPath;
          break;
      }

      const data = await mcpCall("workspace_manager", args);
      setResult(data);
      addHistory({ tool: "workspace_manager", action, label: action, args, result: data });
      onResult?.();
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Workspace operation failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const inputClass = "w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all font-mono text-sm";
  const textareaClass = inputClass + " resize-none min-h-[120px]";

  return (
    <div className="space-y-6">
      <div className="glass rounded-2xl p-6 space-y-5">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-cta/20 rounded-lg">
            <Settings className="w-5 h-5 text-cta" />
          </div>
          <h2 className="text-lg font-bold tracking-tight">Workspace Manager</h2>
        </div>

        <div>
          <label className="block text-xs text-foreground/50 font-medium mb-1.5">Action</label>
          <select
            value={action}
            onChange={(e) => setAction(e.target.value as WorkspaceAction)}
            className={inputClass + " appearance-none cursor-pointer"}
          >
            {actions.map((a) => (
              <option key={a.value} value={a.value}>{a.label}</option>
            ))}
          </select>
        </div>

        {action === "store_context" && (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">Text</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                className={textareaClass}
                placeholder="Architectural decisions or rules to persist..."
              />
            </div>
            <div>
              <label className="block text-xs text-foreground/50 font-medium mb-1.5">Target Path</label>
              <input
                type="text"
                value={targetPath}
                onChange={(e) => setTargetPath(e.target.value)}
                className={inputClass}
                placeholder="e.g. src/components/SearchPanel.tsx"
              />
            </div>
          </div>
        )}

        {action === "delete_context" && (
          <div>
            <label className="block text-xs text-foreground/50 font-medium mb-1.5">Target Path</label>
            <input
              type="text"
              value={targetPath}
              onChange={(e) => setTargetPath(e.target.value)}
              className={inputClass}
              placeholder="e.g. src/components/SearchPanel.tsx"
            />
            <p className="text-xs text-foreground/30 mt-1.5">
              Use <code className="text-cta text-[10px] bg-cta/10 px-1.5 py-0.5 rounded">ALL</code> to delete all stored context entries.
            </p>
          </div>
        )}

        {(action === "get_indexing_diagnostics" || action === "trigger_index") && (
          <p className="text-sm text-foreground/40">No additional inputs required. Click Run to proceed.</p>
        )}

        <button
          onClick={handleRun}
          disabled={loading}
          className="bg-cta text-black font-bold rounded-xl py-2.5 px-6 hover:brightness-110 transition-all flex items-center gap-2 disabled:opacity-40"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {loading ? "Running..." : "Run"}
        </button>
      </div>

      {loading && (
        <div className="glass rounded-2xl p-8 space-y-4">
          <div className="h-4 bg-white/5 rounded animate-pulse" />
          <div className="h-4 bg-white/5 rounded animate-pulse w-3/4" />
          <div className="h-4 bg-white/5 rounded animate-pulse w-1/2" />
        </div>
      )}

      {error && !loading && (
        <div className="glass rounded-2xl p-4 border border-red-500/30 bg-red-500/5">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-red-300 font-mono whitespace-pre-wrap">{error}</p>
            </div>
            <button
              onClick={handleRun}
              className="shrink-0 bg-red-500/20 hover:bg-red-500/30 text-red-300 text-xs font-medium rounded-lg px-3 py-1.5 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {result && !loading && (
        <div className="bg-[#0d1117] rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 border-b border-white/5">
            <span className="text-[11px] font-medium text-foreground/40">Result</span>
            <ResultCopyButton content={result} />
          </div>
          <div className="p-5 overflow-x-auto">
            <MarkdownRenderer content={result} />
          </div>
        </div>
      )}

      {!loading && !result && !error && (
        <div className="glass rounded-2xl p-8 text-center">
          <Settings className="w-8 h-8 text-cta/30 mx-auto mb-3" />
          <p className="text-sm text-foreground/40">Manage workspace context, diagnostics, and re-indexing.</p>
        </div>
      )}
    </div>
  );
}
