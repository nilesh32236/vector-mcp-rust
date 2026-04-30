"use client";

import { useState } from "react";
import { 
  FileCode, 
  ExternalLink, 
  Copy, 
  Check, 
  ChevronDown, 
  ChevronUp,
  Sparkles
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SearchResult {
  id: string;
  text: string;
  similarity: number;
  path: string;
  summary?: string;
  start_line?: number;
  end_line?: number;
}

interface SearchResultCardProps {
  result: SearchResult;
}

export function SearchResultCard({ result }: SearchResultCardProps) {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(result.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const hasSummary = result.summary && result.summary.trim().length > 0;
  const hasLines = result.start_line && result.end_line && result.end_line > 0;

  return (
    <div className="glass p-6 rounded-2xl group hover:border-cta/40 transition-all duration-300">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-primary rounded-xl border border-white/5 group-hover:bg-secondary transition-all">
            <FileCode className="w-5 h-5 text-foreground/70" />
          </div>
          <div>
            <h3 className="font-mono text-sm font-semibold truncate max-w-[200px] sm:max-w-md">
              {result.path || "Manual Context"}
            </h3>
            <p className="text-[10px] font-mono text-foreground/30 uppercase tracking-tighter">
              {hasLines ? `Lines ${result.start_line}–${result.end_line}` : `ID: ${result.id.substring(0, 12)}...`}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden sm:block">
            <p className="text-[10px] uppercase font-bold text-foreground/20 mb-1 tracking-widest">Relevance</p>
            <div className="h-1.5 w-24 bg-primary rounded-full overflow-hidden border border-white/5">
              <div 
                className="h-full bg-cta cta-glow transition-all duration-1000 ease-out" 
                style={{ width: `${(result.similarity ?? 0.8) * 100}%` }}
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button 
              onClick={handleCopy}
              className="p-2 hover:bg-white/5 rounded-lg transition-all cursor-pointer group/btn"
              title="Copy to clipboard"
            >
              {copied ? (
                <Check className="w-4 h-4 text-cta animate-in zoom-in" />
              ) : (
                <Copy className="w-4 h-4 text-foreground/30 group-hover/btn:text-foreground/70" />
              )}
            </button>
            <button className="p-2 hover:bg-white/5 rounded-lg transition-all cursor-pointer group/btn">
              <ExternalLink className="w-4 h-4 text-foreground/30 group-hover/btn:text-foreground/70" />
            </button>
          </div>
        </div>
      </div>

      {/* AI Summary — shown only when available */}
      {hasSummary && (
        <div className="mb-4 flex items-start gap-2.5 px-4 py-3 bg-cta/5 border border-cta/15 rounded-xl">
          <Sparkles className="w-3.5 h-3.5 text-cta mt-0.5 shrink-0" />
          <p className="text-xs text-foreground/70 leading-relaxed italic">
            {result.summary}
          </p>
        </div>
      )}
      
      {/* Code Content */}
      <div className={cn(
        "relative bg-black/30 rounded-xl border border-white/5 overflow-hidden transition-all duration-500",
        isExpanded ? "max-h-[1000px]" : "max-h-32"
      )}>
        <pre className="p-4 font-mono text-xs sm:text-sm text-foreground/80 whitespace-pre-wrap leading-relaxed">
          {result.text}
        </pre>
        
        {!isExpanded && result.text.length > 300 && (
          <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-black/60 to-transparent pointer-events-none" />
        )}
      </div>

      {result.text.length > 300 && (
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-3 flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-foreground/30 hover:text-cta transition-all cursor-pointer mx-auto"
        >
          {isExpanded ? (
            <><ChevronUp className="w-3 h-3" /> Show Less</>
          ) : (
            <><ChevronDown className="w-3 h-3" /> Expand Content</>
          )}
        </button>
      )}
    </div>
  );
}
