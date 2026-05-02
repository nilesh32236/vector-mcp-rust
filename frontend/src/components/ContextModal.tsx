"use client";

import { useState } from "react";

interface ContextModalProps {
  onClose: () => void;
  onSubmit: (text: string, source: string) => Promise<void>;
}

export function ContextModal({ onClose, onSubmit }: ContextModalProps) {
  const [text, setText] = useState("");
  const [source, setSource] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      await onSubmit(text, source);
      onClose(); // only close on success
    } catch (err) {
      setSubmitError(
        err instanceof Error ? err.message : "An unexpected error occurred."
      );
      // Modal stays open — user can retry or cancel
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-md" 
        onClick={onClose}
      />
      <div className="glass w-full max-w-lg rounded-3xl overflow-hidden relative shadow-2xl animate-in fade-in zoom-in duration-300 border-white/20">
        <div className="p-8 border-b border-white/10">
          <h2 className="text-xl font-bold tracking-tight">Add Manual Context</h2>
          <p className="text-sm text-foreground/40 mt-1">Inject external knowledge into the vector store.</p>
        </div>
        <div className="p-8 space-y-6">
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-foreground/60 uppercase tracking-[0.2em]">Source / Reference</label>
            <input 
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="e.g. docs/api-v2.md"
              className="w-full bg-primary/40 border border-white/10 rounded-xl py-3 px-4 outline-none focus:border-cta/50 focus:ring-4 focus:ring-cta/5 transition-all font-mono text-sm"
            />
          </div>
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-foreground/60 uppercase tracking-[0.2em]">Content Payload</label>
            <textarea 
              rows={8}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste raw documentation or code here..."
              className="w-full bg-primary/40 border border-white/10 rounded-xl py-4 px-4 outline-none focus:border-cta/50 focus:ring-4 focus:ring-cta/5 transition-all resize-none font-mono text-sm leading-relaxed"
            />
          </div>
        </div>
        {/* Error message — shown only when submission fails */}
        {submitError && (
          <div className="px-8 pb-2">
            <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3">
              {submitError}
            </p>
          </div>
        )}
        <div className="p-6 bg-white/5 flex justify-end gap-4">
          <button 
            onClick={onClose}
            className="px-5 py-2.5 text-sm font-bold text-foreground/40 hover:text-foreground transition-all cursor-pointer"
          >
            Cancel
          </button>
          <button 
            onClick={handleSubmit}
            disabled={!text || isSubmitting}
            className="px-8 py-2.5 bg-cta text-black font-black rounded-xl text-sm hover:brightness-110 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-xl shadow-cta/20 cta-glow uppercase tracking-widest"
          >
            {isSubmitting ? "Injecting..." : "Inject Context"}
          </button>
        </div>
      </div>
    </div>
  );
}
