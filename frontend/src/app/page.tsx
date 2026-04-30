"use client";

import { useState, useEffect, useCallback } from "react";
import { 
  Search, 
  Settings, 
  Terminal, 
  Database, 
  RefreshCw,
  ChevronRight,
  Command,
  Sparkles
} from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { SearchResultCard } from "@/components/SearchResultCard";
import { ContextModal } from "@/components/ContextModal";

interface SearchResult {
  id: string;
  text: string;
  similarity: number;
  path: string;
  summary?: string;
  start_line?: number;
  end_line?: number;
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
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Load recent searches from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("recent_searches");
    if (saved) setRecentSearches(JSON.parse(saved));
  }, []);

  // Poll status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/tools/status");
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
          setErrorMessage(null);
        } else {
          const errorText = await res.text();
          setErrorMessage(`Failed to fetch status: ${errorText}`);
        }
      } catch (e) {
        console.error("Failed to fetch status", e);
        setErrorMessage("Network error: Failed to fetch status");
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleSearch = useCallback(async (q?: string, save = true) => {
    const searchTerms = q || query;
    if (!searchTerms.trim()) {
      setResults([]);
      return;
    }

    setLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchTerms, top_k: 10 }),
      });
      if (res.ok) {
        const data = await res.json();
        setResults(data);
        
        if (save && data.length > 0) {
          setRecentSearches(prev => {
            const next = [searchTerms, ...prev.filter(s => s !== searchTerms)].slice(0, 5);
            localStorage.setItem("recent_searches", JSON.stringify(next));
            return next;
          });
        }
      } else {
        const errorBody = await res.text();
        setErrorMessage(`Search failed: ${errorBody}`);
      }
    } catch (e) {
      console.error("Search failed", e);
      setErrorMessage("Network error: Search failed");
    } finally {
      setLoading(false);
    }
  }, [query]);

  // Debounced autocomplete/suggestions
  useEffect(() => {
    if (query.length < 3) {
      setSuggestions([]);
      return;
    }

    const timer = setTimeout(() => {
      // In a real app, we'd fetch suggestions here. 
      // For now, we'll just show the query as a suggestion to simulate the UI.
      setSuggestions([`Search for "${query}"`, "Explain the indexing logic", "Show MCP SSE handlers"]);
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        document.getElementById("main-search")?.focus();
      }
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT" && document.activeElement?.tagName !== "TEXTAREA") {
        e.preventDefault();
        document.getElementById("main-search")?.focus();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleAddContext = async (text: string, source: string) => {
    setErrorMessage(null);
    try {
      const res = await fetch("/api/context", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, source }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        setErrorMessage(`Failed to add context: ${errorText}`);
        throw new Error("Failed to add context");
      }
    } catch (e) {
      console.error("Add context failed", e);
      if (!(e instanceof Error)) {
        setErrorMessage("Network error: Failed to add context");
      }
      throw e;
    }
  };

  const handleReindex = async () => {
    setErrorMessage(null);
    try {
      const res = await fetch("/api/tools/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: null }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        setErrorMessage(`Re-index failed: ${errorText}`);
      }
    } catch (e) {
      console.error("Re-index trigger failed", e);
      setErrorMessage("Network error: Re-index trigger failed");
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar 
        status={status} 
        onAddContext={() => setShowContextModal(true)} 
        onReindex={handleReindex}
        recentSearches={recentSearches}
        onSelectRecent={(q) => { setQuery(q); handleSearch(q, false); }}
      />

      <main className="flex-1 flex flex-col relative overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute top-[-15%] right-[-10%] w-[60%] h-[60%] bg-cta/10 blur-[150px] rounded-full pointer-events-none animate-pulse" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-blue-600/5 blur-[120px] rounded-full pointer-events-none" />

        {/* Error Banner */}
        {errorMessage && (
          <div className="mx-10 mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center justify-between animate-in slide-in-from-top-4 duration-300 relative z-20">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              <p className="text-sm text-red-400 font-medium">{errorMessage}</p>
            </div>
            <button 
              onClick={() => setErrorMessage(null)}
              className="text-red-400/50 hover:text-red-400 transition-colors text-xs font-bold uppercase tracking-widest"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Search Header */}
        <header className="p-10 pb-6 relative z-10">
          <div className="max-w-4xl mx-auto">
            <div className="relative group">
              <div className="absolute inset-0 bg-cta/20 blur-2xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-500 rounded-3xl" />
              <div className="relative flex items-center">
                <Search className="absolute left-5 w-6 h-6 text-foreground/30 group-focus-within:text-cta transition-colors duration-300" />
                <input 
                  id="main-search"
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch(query)}
                  placeholder="Query your codebase intelligence..."
                  className="w-full bg-primary/40 border border-white/10 rounded-2xl py-5 pl-14 pr-32 outline-none focus:border-cta/40 transition-all font-sans text-xl glass shadow-2xl placeholder:text-foreground/20"
                />
                <div className="absolute right-5 flex items-center gap-3">
                  {loading && <RefreshCw className="w-5 h-5 animate-spin text-cta" />}
                  <div className="hidden md:flex items-center gap-1.5 px-2 py-1 bg-white/5 rounded-lg border border-white/10">
                    <Command className="w-3 h-3 text-foreground/40" />
                    <span className="text-[10px] font-bold text-foreground/40 tracking-widest">K</span>
                  </div>
                </div>
              </div>

              {/* Suggestions Dropdown */}
              {suggestions.length > 0 && query && (
                <div className="absolute top-full left-0 right-0 mt-2 glass rounded-2xl overflow-hidden shadow-2xl border-white/10 animate-in slide-in-from-top-2 duration-200">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => { 
                        const finalQ = s.includes('"') ? query : s;
                        setQuery(finalQ); 
                        handleSearch(finalQ); 
                        setSuggestions([]); 
                      }}
                      className="w-full text-left px-6 py-4 hover:bg-white/5 flex items-center gap-3 group transition-colors"
                    >
                      <Sparkles className="w-4 h-4 text-cta/40 group-hover:text-cta transition-colors" />
                      <span className="text-sm text-foreground/70 group-hover:text-foreground">{s}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Results Area */}
        <div className="flex-1 overflow-y-auto px-10 pb-10">
          <div className="max-w-4xl mx-auto space-y-6">
            {results.length > 0 ? (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="flex items-center justify-between px-2">
                  <h2 className="text-[10px] font-black uppercase tracking-[0.3em] text-foreground/30">
                    Search Results ({results.length})
                  </h2>
                  <div className="h-px flex-1 bg-white/5 mx-4" />
                </div>
                {results.map((result) => (
                  <SearchResultCard key={result.id} result={result} />
                ))}
              </div>
            ) : query && !loading ? (
              <div className="text-center py-32 animate-in zoom-in duration-300">
                <div className="w-20 h-20 bg-primary/40 rounded-3xl flex items-center justify-center mx-auto mb-6 border border-white/10 shadow-2xl">
                  <Database className="w-10 h-10 text-foreground/10" />
                </div>
                <h3 className="text-xl font-bold text-foreground/60">No matching intel found</h3>
                <p className="text-sm text-foreground/30 mt-2 max-w-xs mx-auto">
                  Try broadening your query or ensure the repository is fully indexed.
                </p>
                <div className="mt-8 flex flex-wrap justify-center gap-2">
                  {["API authentication", "SSE implementation", "Vector storage"].map(t => (
                    <button 
                      key={t}
                      onClick={() => { setQuery(t); handleSearch(t); }}
                      className="px-4 py-2 rounded-full bg-white/5 border border-white/5 text-xs text-foreground/40 hover:border-cta/30 hover:text-cta transition-all"
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            ) : !query && (
              <div className="mt-12">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    { q: "How does the API authentication work?", icon: <Terminal className="w-4 h-4" /> },
                    { q: "Show me the database schema for vectors", icon: <Database className="w-4 h-4" /> },
                    { q: "Where is the MCP SSE logic implemented?", icon: <Settings className="w-4 h-4" /> },
                    { q: "Explain the codebase structure", icon: <Terminal className="w-4 h-4" /> }
                  ].map((item) => (
                    <button 
                      key={item.q}
                      onClick={() => { setQuery(item.q); handleSearch(item.q); }}
                      className="glass-light p-5 rounded-2xl text-left hover:border-cta/40 hover:bg-white/5 transition-all flex items-center justify-between group cursor-pointer"
                    >
                      <div className="flex items-center gap-4">
                        <div className="p-2 bg-primary rounded-lg text-foreground/30 group-hover:text-cta transition-colors">
                          {item.icon}
                        </div>
                        <span className="text-sm font-medium text-foreground/60 group-hover:text-foreground transition-colors">{item.q}</span>
                      </div>
                      <ChevronRight className="w-4 h-4 text-foreground/10 group-hover:text-cta group-hover:translate-x-1 transition-all" />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Modals */}
      {showContextModal && (
        <ContextModal 
          onClose={() => setShowContextModal(false)} 
          onSubmit={handleAddContext} 
        />
      )}
    </div>
  );
}
