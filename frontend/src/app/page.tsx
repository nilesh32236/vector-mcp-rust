"use client";

import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { Code2 } from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { SearchPanel } from "@/components/SearchPanel";
import { AnalyzePanel } from "@/components/AnalyzePanel";
import { ModifyPanel } from "@/components/ModifyPanel";
import { WorkspacePanel } from "@/components/WorkspacePanel";
import { LspPanel } from "@/components/LspPanel";
import { HistoryPanel } from "@/components/HistoryPanel";
import { ContextModal } from "@/components/ContextModal";
import { useKeyboard } from "@/lib/useKeyboard";
import { getHistory, clearHistory } from "@/lib/api";

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState("search");
  const [showContextModal, setShowContextModal] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);

  const refreshHistory = useCallback(() => {
    setHistoryCount(getHistory().length);
  }, []);

  useEffect(() => {
    refreshHistory();
    const interval = setInterval(refreshHistory, 5000);
    return () => clearInterval(interval);
  }, [refreshHistory]);

  const handleAddContext = async (text: string, source: string) => {
    const res = await fetch("/api/context", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source }),
    });
    if (!res.ok) throw new Error(await res.text());
  };

  const handleReindex = async () => {
    toast.info("Re-indexing started…", { id: "reindex" });
    try {
      const res = await fetch("/api/tools/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: null }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        toast.error(`Re-index failed: ${errorText}`, { id: "reindex" });
      }
    } catch {
      toast.error("Network error: Re-index trigger failed", { id: "reindex" });
    }
  };

  useKeyboard([
    { key: "k", meta: true, handler: () => { setActiveTab("search"); setShowHistory(false); }, ignoreInput: true },
    { key: "/", handler: () => { setActiveTab("search"); setShowHistory(false); }, ignoreInput: true },
    { key: "Escape", handler: () => { if (showContextModal) setShowContextModal(false); } },
    ...(["search", "analyze", "modify", "workspace", "lsp"] as const).map((id, i) => ({
      key: String(i + 1),
      meta: true,
      handler: () => { setActiveTab(id); setShowHistory(false); },
    })),
  ]);

  const renderPanel = () => {
    if (showHistory) return <HistoryPanel />;
    switch (activeTab) {
      case "search": return <SearchPanel onResult={() => refreshHistory()} />;
      case "analyze": return <AnalyzePanel onResult={() => refreshHistory()} />;
      case "modify": return <ModifyPanel onResult={() => refreshHistory()} />;
      case "workspace": return <WorkspacePanel onResult={() => refreshHistory()} />;
      case "lsp": return <LspPanel onResult={() => refreshHistory()} />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar
        activeTab={activeTab}
        onTabChange={(tab) => { setActiveTab(tab); setShowHistory(false); }}
        onAddContext={() => setShowContextModal(true)}
        onReindex={handleReindex}
        onClearHistory={() => { clearHistory(); refreshHistory(); }}
        showHistory={showHistory}
        onToggleHistory={() => setShowHistory((v) => !v)}
        historyCount={historyCount}
      />
      <main className="flex-1 flex flex-col relative overflow-hidden">
        <div className="absolute top-[-15%] right-[-10%] w-[60%] h-[60%] bg-cta/10 blur-[150px] rounded-full pointer-events-none animate-pulse" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-blue-600/5 blur-[120px] rounded-full pointer-events-none" />
        <header className="relative z-10 flex items-center gap-3 px-6 py-3 border-b border-white/5">
          <div className="p-1.5 bg-cta/20 rounded-lg">
            <Code2 className="w-4 h-4 text-cta" />
          </div>
          <h1 className="text-sm font-bold tracking-tight">VectorMCP</h1>
          <div className="flex-1" />
          <div className="flex items-center gap-2 text-[10px] text-foreground/30 font-mono">
            <span>Tab {activeTab}</span>
            {historyCount > 0 && (
              <span>{historyCount} history</span>
            )}
          </div>
        </header>
        <div className="relative z-10 flex-1 overflow-y-auto p-6">
          <div key={showHistory ? "history" : activeTab} className="animate-fade-in">
            {renderPanel()}
          </div>
        </div>
      </main>
      {showContextModal && (
        <ContextModal
          onClose={() => setShowContextModal(false)}
          onSubmit={handleAddContext}
        />
      )}
    </div>
  );
}
