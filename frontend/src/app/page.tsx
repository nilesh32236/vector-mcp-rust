"use client";

import { useState, useEffect } from "react";
import { 
  Search, 
  Settings, 
  Terminal, 
  FileCode, 
  Activity, 
  Database, 
  Plus,
  RefreshCw,
  ExternalLink,
  ChevronRight,
  Info
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SearchResult {
  id: string;
  text: string;
  similarity: number;
  path: string;
}

interface IndexStatus {
  status: string;
  indexed_files: number;
  total_files: number;
  current_file: string;
}

export default function WikiPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<IndexStatus | null>(null);
  const [showContextModal, setShowContextModal] = useState(false);
  const [contextText, setContextText] = useState("");
  const [contextSource, setContextSource] = useState("");

  // Poll status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/tools/status");
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
        }
      } catch (e) {
        console.error("Failed to fetch status", e);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 10 }),
      });
      if (res.ok) {
        const data = await res.json();
        setResults(data);
      }
    } catch (e) {
      console.error("Search failed", e);
    } finally {
      setLoading(false);
    }
  };

  const handleAddContext = async () => {
    try {
      const res = await fetch("/api/context", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: contextText, source: contextSource }),
      });
      if (res.ok) {
        setShowContextModal(false);
        setContextText("");
        setContextSource("");
      }
    } catch (e) {
      console.error("Context addition failed", e);
    }
  };

  const handleReindex = async () => {
    try {
      await fetch("/api/tools/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: null }),
      });
      // Status will update via polling
    } catch (e) {
      console.error("Re-index trigger failed", e);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-80 glass border-r flex flex-col">
        <div className="p-6 border-b">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-cta/20 rounded-lg">
              <Terminal className="w-6 h-6 text-cta" />
            </div>
            <h1 className="text-xl font-bold tracking-tight">VectorWiki</h1>
          </div>
          <p className="text-sm text-foreground/60">Local Code Intelligence</p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Status Section */}
          <section>
            <h2 className="text-xs font-semibold text-foreground/40 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Activity className="w-3 h-3" /> System Status
            </h2>
            <div className="space-y-4">
              <div className="glass-light p-4 rounded-xl space-y-3">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-foreground/80">Indexing</span>
                  <span className={cn(
                    "px-2 py-0.5 rounded-full text-xs font-medium",
                    status?.status === "Ready" ? "bg-cta/10 text-cta" : "bg-blue-500/10 text-blue-400"
                  )}>
                    {status?.status || "Idle"}
                  </span>
                </div>
                {status && status.total_files > 0 && (
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs text-foreground/60">
                      <span>{status.indexed_files} / {status.total_files} files</span>
                      <span>{Math.round((status.indexed_files / status.total_files) * 100)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-primary rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-cta transition-all duration-500" 
                        style={{ width: `${(status.indexed_files / status.total_files) * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                {status?.current_file && (
                  <p className="text-[10px] text-foreground/40 truncate italic">
                    {status.current_file}
                  </p>
                )}
              </div>
            </div>
          </section>

          {/* Actions */}
          <section>
            <h2 className="text-xs font-semibold text-foreground/40 uppercase tracking-wider mb-3 flex items-center gap-2">
               Quick Actions
            </h2>
            <button 
              onClick={() => setShowContextModal(true)}
              className="w-full flex items-center gap-3 p-3 rounded-xl hover:bg-white/5 transition-colors text-sm group"
            >
              <div className="p-2 bg-primary rounded-lg group-hover:bg-secondary transition-colors">
                <Plus className="w-4 h-4 text-cta" />
              </div>
              Add Context
            </button>
            <button 
              onClick={handleReindex}
              className="w-full flex items-center gap-3 p-3 rounded-xl hover:bg-white/5 transition-colors text-sm group"
            >
              <div className="p-2 bg-primary rounded-lg group-hover:bg-secondary transition-colors">
                <RefreshCw className={cn("w-4 h-4 text-foreground/60", status?.status !== "Ready" && "animate-spin text-cta")} />
              </div>
              Re-index Repository
            </button>
          </section>
        </div>

        <div className="p-4 border-t glass">
          <div className="flex items-center gap-3 text-xs text-foreground/40">
            <Info className="w-4 h-4" />
            <p>Powered by vector-mcp-rust</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col bg-background/50 relative overflow-hidden">
        {/* Background blobs */}
        <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-cta/5 blur-[120px] rounded-full pointer-events-none" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/5 blur-[100px] rounded-full pointer-events-none" />

        {/* Header / Search */}
        <header className="p-8 pb-4">
          <form onSubmit={handleSearch} className="max-w-3xl mx-auto relative group">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-foreground/30 group-focus-within:text-cta transition-colors" />
            <input 
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything about your codebase..."
              className="w-full bg-primary/50 border border-white/10 rounded-2xl py-4 pl-12 pr-4 outline-none focus:border-cta/50 focus:ring-4 focus:ring-cta/10 transition-all font-sans text-lg glass"
            />
            <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
              {loading && <RefreshCw className="w-4 h-4 animate-spin text-cta" />}
              <kbd className="hidden md:inline-flex h-5 select-none items-center gap-1 rounded border border-white/10 bg-white/5 px-1.5 font-mono text-[10px] font-medium text-foreground/40">
                <span className="text-xs">⌘</span>K
              </kbd>
            </div>
          </form>
        </header>

        {/* Results */}
        <div className="flex-1 overflow-y-auto p-8 pt-4">
          <div className="max-w-4xl mx-auto space-y-6">
            {results.length > 0 ? (
              results.map((result) => (
                <div key={result.id} className="glass p-6 rounded-2xl group hover:border-cta/30 transition-all">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-primary rounded-lg">
                        <FileCode className="w-5 h-5 text-foreground/60" />
                      </div>
                      <div>
                        <h3 className="font-mono text-sm font-semibold truncate max-w-md">
                          {result.path || "Manual Context"}
                        </h3>
                        <p className="text-xs text-foreground/40">ID: {result.id.substring(0, 12)}...</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className="text-[10px] uppercase font-bold text-foreground/30 mb-1">Relevance</p>
                        <div className="h-1.5 w-24 bg-primary rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-cta" 
                            style={{ width: `${(result.similarity || 0.8) * 100}%` }}
                          />
                        </div>
                      </div>
                      <button className="p-2 hover:bg-white/5 rounded-lg transition-colors">
                        <ExternalLink className="w-4 h-4 text-foreground/40" />
                      </button>
                    </div>
                  </div>
                  <div className="bg-black/20 rounded-xl p-4 font-mono text-sm text-foreground/80 overflow-x-auto border border-white/5 whitespace-pre-wrap">
                    {result.text}
                  </div>
                </div>
              ))
            ) : query && !loading ? (
              <div className="text-center py-20">
                <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto mb-4 border border-white/10">
                  <Database className="w-8 h-8 text-foreground/20" />
                </div>
                <h3 className="text-lg font-semibold text-foreground/60">No results found</h3>
                <p className="text-sm text-foreground/40">Try a different query or re-index your repository.</p>
              </div>
            ) : !query && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-12">
                {[
                  "How does the API authentication work?",
                  "Show me the database schema for vectors",
                  "Where is the MCP SSE logic implemented?",
                  "Explain the codebase structure"
                ].map((q) => (
                  <button 
                    key={q}
                    onClick={() => { setQuery(q); handleSearch(); }}
                    className="glass-light p-4 rounded-xl text-left text-sm hover:border-cta/40 transition-all flex items-center justify-between group"
                  >
                    <span className="text-foreground/70">{q}</span>
                    <ChevronRight className="w-4 h-4 text-foreground/20 group-hover:text-cta group-hover:translate-x-1 transition-all" />
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Context Modal */}
      {showContextModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div 
            className="absolute inset-0 bg-black/60 backdrop-blur-sm" 
            onClick={() => setShowContextModal(false)}
          />
          <div className="glass w-full max-w-lg rounded-2xl overflow-hidden relative shadow-2xl animate-in fade-in zoom-in duration-200">
            <div className="p-6 border-b">
              <h2 className="text-lg font-bold">Add Manual Context</h2>
              <p className="text-sm text-foreground/40">Inject external data into the vector store.</p>
            </div>
            <div className="p-6 space-y-4">
              <div className="space-y-2">
                <label className="text-xs font-semibold text-foreground/60 uppercase">Source / Path</label>
                <input 
                  type="text"
                  value={contextSource}
                  onChange={(e) => setContextSource(e.target.value)}
                  placeholder="e.g. documentation/api.md"
                  className="w-full bg-primary/50 border border-white/10 rounded-xl py-2 px-4 outline-none focus:border-cta/50 transition-all"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold text-foreground/60 uppercase">Content</label>
                <textarea 
                  rows={6}
                  value={contextText}
                  onChange={(e) => setContextText(e.target.value)}
                  placeholder="Paste context content here..."
                  className="w-full bg-primary/50 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 transition-all resize-none font-mono text-sm"
                />
              </div>
            </div>
            <div className="p-6 bg-white/5 flex justify-end gap-3">
              <button 
                onClick={() => setShowContextModal(false)}
                className="px-4 py-2 text-sm font-medium hover:text-foreground transition-colors"
              >
                Cancel
              </button>
              <button 
                onClick={handleAddContext}
                disabled={!contextText}
                className="px-6 py-2 bg-cta text-black font-bold rounded-xl text-sm hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-cta/20"
              >
                Inject Context
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
