//! LLM Worker — single-threaded actor that serialises all inference requests.
//!
//! ## Why a dedicated worker thread?
//!
//! `LlamaEngine::summarize_code` allocates a `LlamaContext` and pushes Vulkan
//! compute work on every call.  If the `Scanner` indexes 50 files concurrently,
//! 50 `spawn_blocking` calls would race to allocate contexts simultaneously,
//! exhausting the shared APU VRAM and crashing the Vulkan driver.
//!
//! `LlmWorker` owns the `LlamaEngine` reference and processes requests one at
//! a time through an `mpsc` channel.  No matter how many files are indexed in
//! parallel, only one inference call is in flight at any moment, keeping the
//! memory footprint perfectly stable.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use tokio::sync::{mpsc, oneshot};

use crate::llm::models::LlamaEngine;

/// Maximum time to wait for a single inference request before returning a
/// timeout error to the caller.  Prevents callers from blocking indefinitely
/// when the LLM is under heavy load.
const WORKER_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

/// A single inference request sent to the `LlmWorker`.
pub enum LlmRequest {
    /// Summarise a code chunk and return the result via `reply`.
    Summarize {
        text: String,
        /// Optional streaming channel — each generated token piece is sent here
        /// as it is produced.  `None` means no streaming (batch mode).
        token_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        reply: oneshot::Sender<Result<String>>,
    },
    /// Rerank `candidates` against `query` and return scored indices.
    Rerank {
        query: String,
        candidates: Vec<String>,
        reply: oneshot::Sender<Result<Vec<(usize, f32)>>>,
    },
    /// Clear all on-disk KV-cache files and reset the LRU state.
    ClearKvCache {
        reply: oneshot::Sender<Result<()>>,
    },
    /// Graceful shutdown — the worker loop exits after processing this message.
    #[allow(dead_code)]
    Shutdown,
}

// ---------------------------------------------------------------------------
// LlmWorker
// ---------------------------------------------------------------------------

/// Handle to the single dedicated LLM inference thread.
///
/// Clone freely — all clones share the same underlying channel.
#[derive(Clone)]
pub struct LlmWorker {
    tx: mpsc::Sender<LlmRequest>,
}

impl LlmWorker {
    /// Spawn the worker thread and return a handle.
    ///
    /// The worker runs inside `tokio::task::spawn_blocking` so it never blocks
    /// the async runtime.  It exits when all `LlmWorker` handles are dropped
    /// (the channel closes) or when `shutdown()` is called.
    pub fn spawn(engine: Arc<LlamaEngine>) -> Self {
        let (tx, rx) = mpsc::channel::<LlmRequest>(256);

        tokio::task::spawn_blocking(move || {
            Self::run_loop(engine, rx);
        });

        Self { tx }
    }

    /// The blocking worker loop — runs on a dedicated OS thread.
    fn run_loop(engine: Arc<LlamaEngine>, mut rx: mpsc::Receiver<LlmRequest>) {
        while let Some(req) = rx.blocking_recv() {
            match req {
                LlmRequest::Summarize {
                    text,
                    token_tx,
                    reply,
                } => {
                    let result = engine.summarize_code(&text, token_tx.as_ref());
                    // Ignore send errors — caller may have timed out.
                    let _ = reply.send(result);
                }
                LlmRequest::Rerank {
                    query,
                    candidates,
                    reply,
                } => {
                    let result = engine.rerank_results(&query, &candidates);
                    let _ = reply.send(result);
                }
                LlmRequest::ClearKvCache { reply } => {
                    engine.clear_kv_cache();
                    let _ = reply.send(Ok(()));
                }
                LlmRequest::Shutdown => {
                    tracing::info!("LlmWorker: received Shutdown — exiting run loop");
                    break;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Summarise `text` and return the completed summary.
    ///
    /// Returns `Err` if the worker channel is closed, the worker panicked,
    /// or the request exceeds `WORKER_REQUEST_TIMEOUT`.
    pub async fn summarize(&self, text: String) -> Result<String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(LlmRequest::Summarize {
                text,
                token_tx: None,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow!("LlmWorker channel closed"))?;

        tokio::time::timeout(WORKER_REQUEST_TIMEOUT, reply_rx)
            .await
            .map_err(|_| anyhow!("LlmWorker: summarize timed out after {:?}", WORKER_REQUEST_TIMEOUT))?
            .map_err(|_| anyhow!("LlmWorker dropped reply sender"))?
    }

    /// Summarise `text` with token streaming.
    ///
    /// Each generated token piece is sent on `token_tx` as it is produced.
    /// The completed summary string is returned when generation finishes.
    pub async fn summarize_streaming(
        &self,
        text: String,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(LlmRequest::Summarize {
                text,
                token_tx: Some(token_tx),
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow!("LlmWorker channel closed"))?;

        tokio::time::timeout(WORKER_REQUEST_TIMEOUT, reply_rx)
            .await
            .map_err(|_| anyhow!("LlmWorker: summarize_streaming timed out after {:?}", WORKER_REQUEST_TIMEOUT))?
            .map_err(|_| anyhow!("LlmWorker dropped reply sender"))?
    }

    /// Rerank `candidates` against `query`.
    ///
    /// Returns `(original_index, score)` pairs sorted by descending relevance.
    /// Falls back to identity ranking when the reranker model is not loaded.
    pub async fn rerank(
        &self,
        query: String,
        candidates: Vec<String>,
    ) -> Result<Vec<(usize, f32)>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(LlmRequest::Rerank {
                query,
                candidates,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow!("LlmWorker channel closed"))?;

        tokio::time::timeout(WORKER_REQUEST_TIMEOUT, reply_rx)
            .await
            .map_err(|_| anyhow!("LlmWorker: rerank timed out after {:?}", WORKER_REQUEST_TIMEOUT))?
            .map_err(|_| anyhow!("LlmWorker dropped reply sender"))?
    }

    /// Clear all on-disk KV-cache files via the worker thread.
    pub async fn clear_kv_cache(&self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(LlmRequest::ClearKvCache { reply: reply_tx })
            .await
            .map_err(|_| anyhow!("LlmWorker channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow!("LlmWorker dropped reply sender"))?
    }

    /// Signal the worker to exit cleanly after finishing any in-flight request.
    ///
    /// Returns immediately — the worker exits asynchronously.
    #[allow(dead_code)]
    pub async fn shutdown(&self) {
        let _ = self.tx.send(LlmRequest::Shutdown).await;
    }
}
