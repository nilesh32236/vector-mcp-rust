use anyhow::{anyhow, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use std::num::NonZeroU32;
use std::path::Path;

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
}

impl LlamaEngine {
    /// Load both GGUF models from `models_dir`.
    ///
    /// Both models are fully offloaded to the Vulkan backend via
    /// `n_gpu_layers = 1000`.  Returns `Err` with the missing file path if
    /// either GGUF is absent.
    pub fn new(models_dir: &Path) -> Result<Self> {
        let backend = LlamaBackend::init()
            .map_err(|e| anyhow!("Failed to initialise LlamaBackend: {e}"))?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

        let coder_path = models_dir.join("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
        anyhow::ensure!(
            coder_path.exists(),
            "Coder model not found at: {}",
            coder_path.display()
        );

        let embed_path = models_dir.join("nomic-embed-text-v1.5.Q4_K_M.gguf");
        anyhow::ensure!(
            embed_path.exists(),
            "Embed model not found at: {}",
            embed_path.display()
        );

        let coder_model = LlamaModel::load_from_file(&backend, &coder_path, &model_params)
            .map_err(|e| anyhow!("Failed to load coder model from {}: {e}", coder_path.display()))?;

        let embed_model = LlamaModel::load_from_file(&backend, &embed_path, &model_params)
            .map_err(|e| anyhow!("Failed to load embed model from {}: {e}", embed_path.display()))?;

        Ok(Self {
            _backend: backend,
            coder_model,
            embed_model,
        })
    }

    // -----------------------------------------------------------------------
    // Code summarisation
    // -----------------------------------------------------------------------

    /// Generate a concise one-line summary of `text` using the Qwen2.5-Coder
    /// instruct model.
    ///
    /// A fresh `LlamaContext` is created per call so this method is safe to
    /// call concurrently from multiple threads via `Arc<LlamaEngine>`.
    /// Truncation to the 2048-token context window is handled by llama.cpp
    /// during batch decode — no character-count slicing is performed.
    pub fn summarize_code(&self, text: &str) -> Result<String> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048));

        let mut ctx = self
            .coder_model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create summariser context: {e}"))?;

        // Qwen2.5-Coder instruct chat template
        let prompt = format!(
            "<|im_start|>system\nSummarize the following code concisely in one sentence.<|im_end|>\n\
             <|im_start|>user\n{text}\n<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        let tokens = self
            .coder_model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

        // Prefill batch — only the last token needs logits for sampling
        let mut batch = LlamaBatch::new(2048, 1);
        let last_idx = (tokens.len() - 1) as i32;
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i as i32 == last_idx)
                .map_err(|e| anyhow!("Batch add failed: {e}"))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("Prefill decode failed: {e}"))?;

        // Greedy sampling loop — max 64 new tokens (summaries are short)
        let mut sampler = LlamaSampler::greedy();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let start_pos = batch.n_tokens();

        for (i, _) in (0..64_i32).enumerate() {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if self.coder_model.is_eog_token(token) {
                break;
            }

            sampler.accept(token);

            let piece = self
                .coder_model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| anyhow!("Detokenization failed: {e}"))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, start_pos + i as i32, &[0], true)
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

        const EMBED_CTX: u32 = 1024;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(EMBED_CTX))
            .with_n_batch(EMBED_CTX)
            .with_n_ubatch(EMBED_CTX)
            .with_embeddings(true);

        let mut ctx = self
            .embed_model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create embedding context: {e}"))?;

        let mut tokens = self
            .embed_model
            .str_to_token(&prefixed, AddBos::Always)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;

        // Truncate to context window to avoid "Insufficient Space" / assert failures.
        tokens.truncate(EMBED_CTX as usize);

        let mut batch = LlamaBatch::new(EMBED_CTX as usize, 1);
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
