use crate::llm::models::LlamaEngine;
use anyhow::Result;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Embedder — thin wrapper around Arc<LlamaEngine>
// ---------------------------------------------------------------------------

/// Delegates all embedding work to `LlamaEngine`.
///
/// The old ORT/ONNX/ndarray/tokenizer/hf-hub/turbo-quant stack is removed.
/// `quantize`, `estimate_similarity`, and `rerank` methods are deleted.
pub struct Embedder {
    engine: Option<Arc<LlamaEngine>>,
}

impl Embedder {
    pub fn new(engine: Arc<LlamaEngine>) -> Self {
        Self {
            engine: Some(engine),
        }
    }

    /// Create a disabled embedder that returns `Err` for all operations.
    ///
    /// Used when `LlamaEngine` failed to initialise (e.g. missing model files).
    /// The server starts in degraded mode — search still works via cached
    /// embeddings already stored in LanceDB, but new embeddings cannot be
    /// generated until the engine is available.
    pub fn new_disabled() -> Self {
        Self { engine: None }
    }

    /// Embed a single text for indexing (applies `"search_document: "` prefix).
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("LLM engine not available (degraded mode)"))?;
        let prefixed = if text.starts_with("search_document: ") {
            text.to_string()
        } else {
            format!("search_document: {text}")
        };
        engine.generate_embedding(&prefixed)
    }

    /// Embed a query for search (applies `"search_query: "` prefix).
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("LLM engine not available (degraded mode)"))?;
        let prefixed = if text.starts_with("search_query: ") {
            text.to_string()
        } else {
            format!("search_query: {text}")
        };
        engine.generate_embedding(&prefixed)
    }

    /// Embed a batch of texts sequentially (one vector per input string).
    ///
    /// # Why sequential?
    /// `LlamaEngine::generate_embedding` allocates a `LlamaContext` and submits
    /// Vulkan compute work on each call. Parallelising with rayon would cause
    /// multiple contexts to contend for the shared APU VRAM simultaneously,
    /// exhausting memory and hanging the Vulkan driver. Sequential processing
    /// keeps the memory footprint stable regardless of batch size.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_text(t)).collect()
    }
}
