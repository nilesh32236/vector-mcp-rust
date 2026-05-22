"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";

interface ResultCopyButtonProps {
  content: string;
  label?: string;
}

export function ResultCopyButton({ content, label = "Copy" }: ResultCopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // silent
    }
  };

  return (
    <button
      onClick={handleCopy}
      aria-label="Copy result"
      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition-all cursor-pointer"
      style={{
        color: copied ? "#2f9e44" : "rgba(255,255,255,0.4)",
        background: copied ? "rgba(47,158,68,0.1)" : "rgba(255,255,255,0.04)",
      }}
      onMouseEnter={(e) => {
        if (!copied) e.currentTarget.style.background = "rgba(255,255,255,0.08)";
      }}
      onMouseLeave={(e) => {
        if (!copied) e.currentTarget.style.background = "rgba(255,255,255,0.04)";
      }}
    >
      {copied ? (
        <Check className="w-3 h-3" />
      ) : (
        <Copy className="w-3 h-3" />
      )}
      <span className="hidden sm:inline">{copied ? "Copied" : label}</span>
    </button>
  );
}
