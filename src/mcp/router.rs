//! Semantic intent router — classifies query intent before dispatching to the LLM.
//!
//! Uses cosine similarity between the query embedding and a small set of
//! pre-computed prototype vectors to classify intent in <5ms on the Ryzen APU.
//! Falls back to `SearchOnly` when the embedder is in degraded mode.

use crate::llm::embedding::Embedder;

// ---------------------------------------------------------------------------
// Intent enum
// ---------------------------------------------------------------------------

/// Classified intent of a user query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    /// Pure semantic/keyword search — no LLM generation needed.
    SearchOnly = 0,
    /// Requires Qwen2.5-Coder to generate a summary or explanation.
    Summarize = 1,
    /// Requires structural analysis (dead code, architecture, dependencies).
    Analyze = 2,
}

// ---------------------------------------------------------------------------
// SemanticRouter
// ---------------------------------------------------------------------------

/// Lightweight intent classifier backed by prototype vector similarity.
pub struct SemanticRouter {
    /// Pre-computed L2-normalised prototype vectors per intent.
    /// Both vectors are already normalised by `generate_embedding`, so
    /// cosine similarity = dot product.
    prototypes: Vec<(Intent, Vec<f32>)>,
}

/// Representative phrases for each intent class.
/// These are embedded once at startup and stored as prototype vectors.
const PROTOTYPE_PHRASES: &[(&str, Intent)] = &[
    // SearchOnly
    ("find function", Intent::SearchOnly),
    ("search for", Intent::SearchOnly),
    ("where is", Intent::SearchOnly),
    ("show me", Intent::SearchOnly),
    ("list all", Intent::SearchOnly),
    ("locate", Intent::SearchOnly),
    // Summarize
    ("explain", Intent::Summarize),
    ("summarize", Intent::Summarize),
    ("what does this do", Intent::Summarize),
    ("describe", Intent::Summarize),
    ("how does", Intent::Summarize),
    ("document this", Intent::Summarize),
    // Analyze
    ("dead code", Intent::Analyze),
    ("architecture", Intent::Analyze),
    ("dependencies", Intent::Analyze),
    ("duplicate code", Intent::Analyze),
    ("refactor", Intent::Analyze),
    ("analyze", Intent::Analyze),
];

impl SemanticRouter {
    /// Build the router by embedding all prototype phrases.
    ///
    /// Called once at startup after the `Embedder` is ready.
    /// Phrases that fail to embed (e.g. degraded mode) are silently skipped.
    pub fn build(embedder: &Embedder) -> Self {
        let prototypes: Vec<(Intent, Vec<f32>)> = PROTOTYPE_PHRASES
            .iter()
            .filter_map(|(phrase, intent)| {
                embedder
                    .embed_query(phrase)
                    .ok()
                    .map(|v| (*intent, v))
            })
            .collect();

        tracing::info!(
            attempted = PROTOTYPE_PHRASES.len(),
            successful = prototypes.len(),
            "SemanticRouter: prototype vectors built"
        );

        Self { prototypes }
    }

    /// Classify the intent of `query`.
    ///
    /// Aggregates cosine similarity scores per intent class and returns the
    /// class with the highest total score. Falls back to `SearchOnly` when
    /// the embedder is unavailable or no prototypes were built.
    pub fn classify(&self, query: &str, embedder: &Embedder) -> Intent {
        if self.prototypes.is_empty() {
            return Intent::SearchOnly;
        }

        let query_vec = match embedder.embed_query(query) {
            Ok(v) => v,
            Err(_) => return Intent::SearchOnly,
        };

        // Aggregate dot-product scores per intent (vectors are L2-normalised).
        // Use per-intent averages to avoid bias from unequal prototype counts.
        let mut search_score = 0.0_f32;
        let mut summarize_score = 0.0_f32;
        let mut analyze_score = 0.0_f32;
        let mut search_count = 0u32;
        let mut summarize_count = 0u32;
        let mut analyze_count = 0u32;

        for (intent, proto) in &self.prototypes {
            let sim = dot_product(&query_vec, proto);
            match intent {
                Intent::SearchOnly => {
                    search_score += sim;
                    search_count += 1;
                }
                Intent::Summarize => {
                    summarize_score += sim;
                    summarize_count += 1;
                }
                Intent::Analyze => {
                    analyze_score += sim;
                    analyze_count += 1;
                }
            }
        }

        // Normalise to per-intent averages.
        let search_avg = if search_count > 0 { search_score / search_count as f32 } else { 0.0 };
        let summarize_avg = if summarize_count > 0 { summarize_score / summarize_count as f32 } else { 0.0 };
        let analyze_avg = if analyze_count > 0 { analyze_score / analyze_count as f32 } else { 0.0 };

        let intent = if summarize_avg >= search_avg && summarize_avg >= analyze_avg {
            Intent::Summarize
        } else if analyze_avg >= search_avg && analyze_avg >= summarize_avg {
            Intent::Analyze
        } else {
            Intent::SearchOnly
        };

        tracing::debug!(
            query = %query,
            intent = ?intent,
            search_avg,
            summarize_avg,
            analyze_avg,
            "SemanticRouter: classified intent"
        );

        intent
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dot product of two equal-length slices.
/// For L2-normalised vectors this equals cosine similarity.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_product: slice lengths must be equal");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
