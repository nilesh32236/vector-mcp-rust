use crate::config::Config;
use crate::llm::models::LlamaEngine;
use anyhow::{Context, Result};
use dashmap::DashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Summarizer — thin wrapper around Arc<LlamaEngine> with SHA-256 cache
// ---------------------------------------------------------------------------

/// Delegates all generation work to `LlamaEngine`.
///
/// The old candle-core / tokenizer / hf-hub / turbo-quant / semantic-cache
/// stack is removed.  Only the SHA-256 exact-match cache is retained.
pub struct Summarizer {
    engine: Option<Arc<LlamaEngine>>,
    /// Exact-match cache: SHA-256(content) → summary string.
    cache: Arc<DashMap<String, String>>,
}

impl Summarizer {
    /// Create a new `Summarizer`.
    ///
    /// Pass `Some(engine)` when `enable_local_llm` is `true`; pass `None`
    /// to create a no-op summarizer that always returns `"LLM Disabled"`.
    pub fn new(engine: Option<Arc<LlamaEngine>>) -> Self {
        Self {
            engine,
            cache: Arc::new(DashMap::new()),
        }
    }

    /// Summarize a code chunk.
    ///
    /// Returns `Ok("LLM Disabled")` when `enable_local_llm` is `false`.
    /// Returns the cached summary on a SHA-256 hit without invoking the engine.
    /// Offloads inference to `spawn_blocking` to avoid blocking the async runtime.
    pub async fn summarize_chunk(&self, text: &str, config: Arc<Config>) -> Result<String> {
        if !config.feature_toggles.enable_local_llm {
            return Ok("LLM Disabled".to_string());
        }

        let hash = sha256_hex(text);

        if let Some(cached) = self.cache.get(&hash) {
            tracing::debug!("Summary cache hit");
            return Ok(cached.clone());
        }

        let engine = self
            .engine
            .as_ref()
            .context("LlamaEngine not loaded (enable_local_llm is true but engine is None)")?;

        let summary = tokio::task::spawn_blocking({
            let engine = Arc::clone(engine);
            let text = text.to_string();
            move || engine.summarize_code(&text)
        })
        .await
        .context("spawn_blocking panicked")??;

        self.cache.insert(hash, summary.clone());
        Ok(summary)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sha256_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    hex::encode(h.finalize())
}
