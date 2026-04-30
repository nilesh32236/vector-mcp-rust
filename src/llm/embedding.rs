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
    engine: Arc<LlamaEngine>,
}

impl Embedder {
    pub fn new(engine: Arc<LlamaEngine>) -> Self {
        Self { engine }
    }

    /// Embed a single text for indexing (applies `"search_document: "` prefix).
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.engine.generate_embedding(text)
    }

    /// Embed a query for search (applies `"search_query: "` prefix).
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = if text.starts_with("search_query: ") {
            text.to_string()
        } else {
            format!("search_query: {text}")
        };
        self.engine.generate_embedding(&prefixed)
    }

    /// Embed a batch of texts (one vector per input string).
    #[allow(dead_code)]
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|t| self.engine.generate_embedding(t))
            .collect()
    }
}
