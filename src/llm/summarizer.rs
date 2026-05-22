use crate::config::Config;
use crate::llm::worker::LlmWorker;
use anyhow::{Context, Result};
use dashmap::DashMap;
use sha2::{Digest, Sha256};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Summarizer — thin wrapper around LlmWorker with SHA-256 cache
// ---------------------------------------------------------------------------

/// Delegates all generation work to `LlmWorker`.
///
/// The old candle-core / tokenizer / hf-hub / turbo-quant / semantic-cache
/// stack is removed.  Only the SHA-256 exact-match cache is retained.
///
/// `Summarizer` is wrapped in `Arc<Summarizer>` at every call site, so the
/// `cache` field does not need its own `Arc` wrapper — `DashMap` is `Sync`
/// and safe to access via `&self`.
pub struct Summarizer {
    worker: Option<Arc<LlmWorker>>,
    /// Exact-match cache: SHA-256(content) → summary string.
    cache: DashMap<String, String>,
}

impl Summarizer {
    /// Create a new `Summarizer`.
    ///
    /// Pass `Some(worker)` when `enable_local_llm` is `true`; pass `None`
    /// to create a no-op summarizer that always returns `"LLM Disabled"`.
    pub fn new(worker: Option<Arc<LlmWorker>>) -> Self {
        Self {
            worker,
            cache: DashMap::new(),
        }
    }

    /// Summarize a code chunk.
    ///
    /// Returns `Ok("LLM Disabled")` when `enable_local_llm` is `false`.
    /// Returns the cached summary on a SHA-256 hit without invoking the worker.
    /// Sends the request to the `LlmWorker` channel for sequential processing.
    pub async fn summarize_chunk(&self, text: &str, config: Arc<Config>) -> Result<String> {
        if !config.feature_toggles.enable_local_llm {
            return Ok("LLM Disabled".to_string());
        }

        let hash = sha256_hex(text);

        if let Some(cached) = self.cache.get(&hash) {
            tracing::debug!("Summary cache hit");
            return Ok(cached.clone());
        }

        let worker = self
            .worker
            .as_ref()
            .context("LlmWorker not available (enable_local_llm is true but worker is None)")?;

        let summary = worker
            .summarize(text.to_string())
            .await
            .context("LlmWorker summarize failed")?;

        self.cache.insert(hash, summary.clone());
        Ok(summary)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sha256_hex(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    hex::encode(h.finalize())
}
