"use client";

import { 
  Terminal, 
  Activity, 
  Plus, 
  RefreshCw, 
  Info,
  Clock
} from "lucide-react";
import { cn } from "@/lib/utils";

interface IndexStatus {
  status: string;
  indexed_files: number;
  total_files: number;
  current_file: string;
}

interface SidebarProps {
  status: IndexStatus | null;
  onAddContext: () => void;
  onReindex: () => void;
  recentSearches?: string[];
  onSelectRecent?: (query: string) => void;
}

export function Sidebar({ 
  status, 
  onAddContext, 
  onReindex, 
  recentSearches = [], 
  onSelectRecent 
}: SidebarProps) {
  return (
    <aside className="w-80 glass border-r flex flex-col h-full">
      <div className="p-6 border-b">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-cta/20 rounded-lg cta-glow">
            <Terminal className="w-6 h-6 text-cta" />
          </div>
          <h1 className="text-xl font-bold tracking-tight">VectorWiki</h1>
        </div>
        <p className="text-sm text-foreground/60">Local Code Intelligence</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Status Section */}
        <section>
          <h2 className="text-[10px] font-bold text-foreground/40 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
            <Activity className="w-3 h-3" /> System Status
          </h2>
          <div className="space-y-4">
            <div className="glass-light p-4 rounded-xl space-y-3">
              <div className="flex justify-between items-center text-sm">
                <span className="text-foreground/80 font-medium">Indexing</span>
                <span className={cn(
                  "px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider",
                  status?.status === "Ready" ? "bg-cta/20 text-cta" : "bg-blue-500/20 text-blue-400"
                )}>
                  {status?.status || "Idle"}
                </span>
              </div>
              {status && status.total_files > 0 && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-[10px] font-mono text-foreground/50">
                    <span>{status.indexed_files} / {status.total_files} files</span>
                    <span>{Math.round((status.indexed_files / status.total_files) * 100)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-primary rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-cta transition-all duration-700 ease-out" 
                      style={{ width: `${(status.indexed_files / status.total_files) * 100}%` }}
                    />
                  </div>
                </div>
              )}
              {status?.current_file && (
                <p className="text-[10px] text-foreground/40 truncate italic font-mono">
                  {status.current_file}
                </p>
              )}
            </div>
          </div>
        </section>

        {/* Recent Searches */}
        {recentSearches.length > 0 && (
          <section>
            <h2 className="text-[10px] font-bold text-foreground/40 uppercase tracking-[0.2em] mb-3 flex items-center gap-2">
              <Clock className="w-3 h-3" /> Recent Searches
            </h2>
            <div className="space-y-1">
              {recentSearches.map((search, i) => (
                <button
                  key={i}
                  onClick={() => onSelectRecent?.(search)}
                  className="w-full text-left p-2 rounded-lg text-xs text-foreground/60 hover:bg-white/5 hover:text-foreground transition-all truncate"
                >
                  {search}
                </button>
              ))}
            </div>
          </section>
        )}

        {/* Actions */}
        <section>
          <h2 className="text-[10px] font-bold text-foreground/40 uppercase tracking-[0.2em] mb-3 flex items-center gap-2">
             Quick Actions
          </h2>
          <div className="space-y-1">
            <button 
              onClick={onAddContext}
              className="w-full flex items-center gap-3 p-3 rounded-xl hover:bg-white/5 transition-all text-sm group cursor-pointer"
            >
              <div className="p-2 bg-primary rounded-lg group-hover:bg-secondary transition-all">
                <Plus className="w-4 h-4 text-cta" />
              </div>
              <span className="font-medium">Add Context</span>
            </button>
            <button 
              onClick={onReindex}
              className="w-full flex items-center gap-3 p-3 rounded-xl hover:bg-white/5 transition-all text-sm group cursor-pointer"
            >
              <div className="p-2 bg-primary rounded-lg group-hover:bg-secondary transition-all">
                <RefreshCw className={cn("w-4 h-4 text-foreground/60", status?.status !== "Ready" && "animate-spin text-cta")} />
              </div>
              <span className="font-medium">Re-index Repository</span>
            </button>
          </div>
        </section>
      </div>

      <div className="p-4 border-t glass">
        <div className="flex items-center gap-3 text-[10px] text-foreground/40 font-mono">
          <Info className="w-4 h-4" />
          <p>VERSION 0.1.0-ALPHA</p>
        </div>
      </div>
    </aside>
  );
}
