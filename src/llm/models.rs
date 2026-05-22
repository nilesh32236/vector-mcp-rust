use anyhow::{anyhow, Result};
use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use sha2::{Digest, Sha256};
use std::num::NonZeroU32;
use std::path::Path;

use crate::llm::kv_cache::KvCacheStore;

// ---------------------------------------------------------------------------
// LlamaEngine — unified inference engine for embeddings + summarisation
// ---------------------------------------------------------------------------

/// Owns the llama.cpp backend singleton and both loaded GGUF models.
///
/// `LlamaEngine` is `Send + Sync`: `LlamaBackend` and `LlamaModel` are both
/// `Send + Sync` in llama-cpp-2; `LlamaContext` is created per-call so there
/// is no shared mutable state.  Wrap in `Arc<LlamaEngine>` for sharing across
/// async tasks.
pub struct LlamaEngine {
    /// Keeps the backend alive for the process lifetime (must outlive models).
    _backend: LlamaBackend,
    /// Qwen2.5-Coder 0.5B Q4_K_M — used for code summarisation.
    coder_model: LlamaModel,
    /// nomic-embed-text-v1.5 Q4_K_M — used for vector embeddings.
    embed_model: LlamaModel,
    /// Optional reranker model (e.g. bge-reranker-v2-m3-Q4_K_M).
    reranker_model: Option<LlamaModel>,
    /// On-disk KV-cache store for zero-latency follow-up summarisations.
    kv_cache: Option<KvCacheStore>,
}

impl LlamaEngine {
    /// Load both GGUF models from `models_dir`.
    ///
    /// Both models are fully offloaded to the Vulkan backend via
    /// `n_gpu_layers = 1000`.  Returns `Err` with the missing file path if
    /// either GGUF is absent.
    #[allow(dead_code)]
    pub fn new(models_dir: &Path) -> Result<Self> {
        Self::new_with_config(models_dir, None, None, None, None)
    }

    /// Full constructor — used by `init_components` to pass KV-cache and
    /// reranker configuration from `Config`.
    pub fn new_with_config(
        models_dir: &Path,
        kv_cache_dir: Option<&Path>,
        kv_cache_max_entries: Option<usize>,
        reranker_model_path: Option<&Path>,
        embed_model_path: Option<&Path>,
    ) -> Result<Self> {
        let backend = LlamaBackend::init()
            .map_err(|e| anyhow!("Failed to initialise LlamaBackend: {e}"))?;

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(1000)
            .with_use_mmap(true)
            .with_use_mlock(true);
        tracing::info!(
            "LlamaModelParams: n_gpu_layers=1000, use_mmap=true, use_mlock=true. \
             Note: mlock may silently fail if RLIMIT_MEMLOCK is insufficient — \
             add LimitMEMLOCK=infinity to the systemd unit to guarantee locking."
        );

        let mut coder_path = models_dir.join("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
        if !coder_path.exists() {
            if let Ok(home) = std::env::var("HOME") {
                let fallback = Path::new(&home).join(".local/share/vector-mcp-rust/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
                if fallback.exists() {
                    coder_path = fallback;
                }
            }
        }

        anyhow::ensure!(
            coder_path.exists(),
            "Coder model not found at: {}",
            coder_path.display()
        );

        let embed_path = embed_model_path.map(std::path::PathBuf::from).unwrap_or_else(|| {
            models_dir.join("e5-small-v2.Q8_0.gguf")
        });
        anyhow::ensure!(
            embed_path.exists(),
            "Embed model not found at: {}",
            embed_path.display()
        );

        let coder_model = LlamaModel::load_from_file(&backend, &coder_path, &model_params)
            .map_err(|e| anyhow!("Failed to load coder model from {}: {e}", coder_path.display()))?;

        let embed_model = LlamaModel::load_from_file(&backend, &embed_path, &model_params)
            .map_err(|e| anyhow!("Failed to load embed model from {}: {e}", embed_path.display()))?;

        // Optionally load the reranker model — non-fatal if absent.
        let reranker_model = reranker_model_path.and_then(|p| {
            if p.exists() {
                match LlamaModel::load_from_file(&backend, p, &model_params) {
                    Ok(m) => {
                        tracing::info!(path = %p.display(), "Reranker model loaded");
                        Some(m)
                    }
                    Err(e) => {
                        tracing::warn!(path = %p.display(), error = %e, "Failed to load reranker model — reranking disabled");
                        None
                    }
                }
            } else {
                tracing::info!(path = %p.display(), "Reranker model not found — reranking disabled");
                None
            }
        });

        // Initialise KV-cache store if a directory is configured.
        // Purge any files left over from a previous run — they may have been
        // saved with a different n_ctx and would cause SIGABRT if loaded.
        let kv_cache = kv_cache_dir.map(|dir| {
            let store = KvCacheStore::new(dir.to_path_buf(), kv_cache_max_entries.unwrap_or(10));
            store.clear_on_startup();
            store
        });

        Ok(Self {
            _backend: backend,
            coder_model,
            embed_model,
            reranker_model,
            kv_cache,
        })
    }

    // -----------------------------------------------------------------------
    // Code summarisation
    // -----------------------------------------------------------------------

    /// Generate a concise one-line summary of `text` using the Qwen2.5-Coder
    /// instruct model.
    ///
    /// A fresh `LlamaContext` is created per call so this method is safe to
    /// call from the dedicated `LlmWorker` thread.
    ///
    /// If `token_tx` is `Some`, each generated token piece is sent on the
    /// channel as it is produced, enabling real-time SSE streaming to the
    /// frontend.  When `token_tx` is `None`, behaviour is identical to the
    /// non-streaming path — no allocation overhead.
    pub fn summarize_code(
        &self,
        text: &str,
        token_tx: Option<&tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> Result<String> {
        // Qwen2.5-Coder instruct chat template
        let prompt = format!(
            "<|im_start|>system\nSummarize the following code concisely in one sentence.<|im_end|>\n\
             <|im_start|>user\n{text}\n<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        // Tokenize first to determine the minimum context size needed.
        let tokens = self
            .coder_model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

        // Dynamic n_ctx: next power-of-two >= token_count + 64 (generation budget), capped at 2048.
        let n_ctx = next_pow2_ctx(tokens.len() + 64, 512, 2048);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx));

        let mut ctx = self
            .coder_model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create summariser context: {e}"))?;

        // Compute prompt hash for KV-cache lookup.
        // NOTE: KV-cache state persistence is disabled because state_load_file
        // in llama-cpp-2 v0.1.145 produces corrupt output buffers when the
        // saved context n_ctx differs from the current context n_ctx, causing
        // GGML_ASSERT(logits != nullptr) → SIGABRT. The feature is kept as
        // dead code so it can be re-enabled when the upstream bug is fixed.
        let _prompt_hash = sha256_hex(&prompt);
        let kv_cache_hit = false;

        // Full prefill — always performed (KV-cache load disabled).
        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        if !kv_cache_hit {
            let last_idx = (tokens.len() - 1) as i32;
            for (i, &token) in tokens.iter().enumerate() {
                batch
                    .add(token, i as i32, &[0], i as i32 == last_idx)
                    .map_err(|e| anyhow!("Batch add failed: {e}"))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("Prefill decode failed: {e}"))?;

            // KV-cache save disabled — see note above about state_load_file crash.
        }

        // Greedy sampling loop — max 64 new tokens (summaries are short)
        let mut sampler = LlamaSampler::greedy();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let start_pos = batch.n_tokens();

        for i in 0..64_i32 {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if self.coder_model.is_eog_token(token) {
                break;
            }

            sampler.accept(token);

            let piece = self
                .coder_model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| anyhow!("Detokenization failed: {e}"))?;

            // Stream token to SSE client if a channel is provided.
            if let Some(tx) = token_tx {
                let _ = tx.send(piece.clone());
            }

            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, start_pos + i, &[0], true)
                .map_err(|e| anyhow!("Batch add (generation) failed: {e}"))?;
            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("Generation decode failed: {e}"))?;
        }

        Ok(output.trim().to_string())
    }

    // -----------------------------------------------------------------------
    // Vector embeddings
    // -----------------------------------------------------------------------

    /// Generate an L2-normalised embedding vector for `text` using the
    /// nomic-embed-text-v1.5 model.
    ///
    /// The `"search_document: "` prefix required by nomic-embed-text-v1.5 is
    /// applied idempotently — it is skipped if the text already starts with
    /// `"search_document: "` or `"search_query: "`.
    ///
    /// Returns `Err` for empty input.
    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(anyhow!("generate_embedding: empty input is not supported"));
        }

        // Apply nomic-embed prefix idempotently
        let prefixed: String =
            if text.starts_with("search_document: ") || text.starts_with("search_query: ") {
                text.to_string()
            } else {
                format!("search_document: {text}")
            };

        // Tokenize first to determine the minimum context size needed.
        let mut tokens = self
            .embed_model
            .str_to_token(&prefixed, AddBos::Always)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

        // Truncate to hard cap to avoid "Insufficient Space" / assert failures.
        tokens.truncate(1024);
        let token_count = tokens.len();

        // Dynamic n_ctx: next power-of-two >= token_count, floored at 64.
        // Cap at the model's training context (e.g. 512 for BGE-small) or 1024.
        let n_ctx_train = self.embed_model.n_ctx_train();
        let n_ctx = (token_count as u32)
            .next_power_of_two()
            .max(64)
            .min(n_ctx_train)
            .min(1024);

        // Crucial: Truncate tokens to the context size we actually allocated
        // to avoid "Insufficient Space" errors in LlamaBatch.
        tokens.truncate(n_ctx as usize);

        // Only force Mean pooling if the model doesn't have internal pooling.
        // For models like Jina, this is required. For others, it might override 
        // to Mean instead of CLS, which is acceptable for benchmarking.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_ubatch(n_ctx)
            .with_pooling_type(LlamaPoolingType::Mean)
            .with_embeddings(true);

        let mut ctx = self
            .embed_model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create embedding context: {e}"))?;

        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        batch
            .add_sequence(&tokens, 0, false)
            .map_err(|e| anyhow!("Batch add_sequence failed: {e}"))?;

        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("Embedding decode failed: {e}"))?;

        let raw = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow!("Failed to extract embeddings: {e}"))?;

        Ok(l2_normalize(raw.to_vec()))
    }

    /// Clear the on-disk KV-cache (called from `LlmWorker`).
    pub fn clear_kv_cache(&self) {
        if let Some(ref kv) = self.kv_cache {
            kv.clear();
        }
    }

    // -----------------------------------------------------------------------
    // Cross-encoder reranking
    // -----------------------------------------------------------------------

    /// Rerank `candidates` against `query` using the cross-encoder reranker model.
    ///
    /// Returns `(original_index, score)` pairs sorted by descending relevance.
    ///
    /// When no reranker model is loaded, returns an identity ranking
    /// `[(0, 0.0), (1, 0.0), ...]` so callers can always use the result
    /// without special-casing the absent-model path.
    pub fn rerank_results(
        &self,
        query: &str,
        candidates: &[String],
    ) -> Result<Vec<(usize, f32)>> {
        if self.reranker_model.is_none() {
            tracing::debug!("rerank_results: no reranker model loaded — returning identity ranking");
            return Ok((0..candidates.len()).map(|i| (i, 0.0_f32)).collect());
        }

        // Score each (query, candidate) pair sequentially to avoid VRAM contention.
        let mut scores: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, candidate)| {
                let score = self
                    .score_pair(query, candidate)
                    .unwrap_or(0.0);
                (i, score)
            })
            .collect();

        // Sort descending by score.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores)
    }

    /// Score a single (query, document) pair using the reranker model.
    ///
    /// Uses the standard cross-encoder input format: `"query [SEP] document"`.
    /// Extracts the first logit as the relevance score.
    fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        let reranker = self
            .reranker_model
            .as_ref()
            .ok_or_else(|| anyhow!("Reranker model not loaded"))?;

        let input = format!("{query} [SEP] {document}");

        let tokens = reranker
            .str_to_token(&input, AddBos::Always)
            .map_err(|e| anyhow!("Reranker tokenization failed: {e}"))?;

        let token_count = tokens.len().min(512);
        let n_ctx = (token_count as u32).next_power_of_two().max(64).min(512);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_embeddings(true);

        let mut ctx = reranker
            .new_context(&self._backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create reranker context: {e}"))?;

        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        let last_idx = (token_count - 1) as i32;
        for (i, &token) in tokens[..token_count].iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i as i32 == last_idx)
                .map_err(|e| anyhow!("Reranker batch add failed: {e}"))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("Reranker decode failed: {e}"))?;

        // Extract the relevance score from the sequence embedding.
        // For ranking models like bge-reranker, llama.cpp exposes the
        // sequence classification score as the sequence embedding.
        let score = ctx
            .embeddings_seq_ith(0)
            .ok()
            .and_then(|e| e.first().copied())
            .unwrap_or(0.0);

        Ok(score)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// Compute the smallest power-of-two context size that is:
/// - at least `floor`
/// - at least `min` tokens
/// - at most `cap`
///
/// Used for dynamic context sizing: allocate only as much VRAM as the prompt
/// actually needs, leaving headroom for other operations on the shared APU.
///
/// # Examples
/// ```
/// assert_eq!(next_pow2_ctx(100, 512, 2048), 512);  // small chunk → floor
/// assert_eq!(next_pow2_ctx(600, 512, 2048), 1024); // medium chunk → 1024
/// assert_eq!(next_pow2_ctx(2000, 512, 2048), 2048); // large chunk → cap
/// ```
fn next_pow2_ctx(min: usize, floor: u32, cap: u32) -> u32 {
    if min <= floor as usize {
        return floor;
    }
    let mut n = floor;
    while (n as usize) < min {
        n = n.saturating_mul(2);
        if n >= cap {
            return cap;
        }
    }
    n.min(cap)
}

/// Compute the SHA-256 hex digest of `text`.
///
/// Used to key KV-cache files so that identical prompts reuse the same
/// cached context state.
fn sha256_hex(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    hex::encode(h.finalize())
}
