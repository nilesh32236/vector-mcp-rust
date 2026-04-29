use crate::config::Config;
use crate::llm::util::download_direct;
use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use hf_hub::api::sync::Api;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

pub struct Summarizer {
    model: Option<Mutex<ModelWeights>>,
    tokenizer: Option<Tokenizer>,
    quantizer: Option<turbo_quant::TurboQuantizer>,
    cache: Arc<dashmap::DashMap<String, String>>, // Exact match cache: content_hash -> summary
    semantic_cache: Arc<dashmap::DashMap<Vec<u8>, String>>, // Semantic cache: TurboCode -> summary
}

impl Summarizer {
    /// Create a new Summarizer and load the model if local LLM is enabled.
    pub async fn new(config: &Config) -> Result<Self> {
        let quantizer = if config.dimension > 0 && config.dimension % 2 == 0 {
            turbo_quant::TurboQuantizer::new(config.dimension, 8, config.dimension / 4, 42).ok()
        } else {
            None
        };

        if !config.feature_toggles.enable_local_llm {
            return Ok(Self {
                model: None,
                tokenizer: None,
                quantizer,
                cache: Arc::new(dashmap::DashMap::new()),
                semantic_cache: Arc::new(dashmap::DashMap::new()),
            });
        }

        // 1. Download/Cache model from Hugging Face
        // For optimal performance, always use 4-bit quantized GGUF models (e.g., Q4_K_M)
        let local_model = config.models_dir.join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
        let model_path = if local_model.exists() {
            tracing::info!("Found local model at {:?}", local_model);
            local_model
        } else {
            tracing::info!(
                "Local model not found in models_dir, provisioning (using RAM-efficient Q4_K_M quantization)..."
            );
            std::fs::create_dir_all(&config.models_dir)?;

            // Try direct download first for reliability
            let qwen_gguf_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf";
            if let Err(e) = download_direct(qwen_gguf_url, &local_model).await {
                tracing::warn!(
                    "Direct download of Qwen GGUF failed: {}. Falling back to HF Hub...",
                    e
                );
                let api = Api::new().context("Failed to create HF API client")?;
                let repo = api.model("Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string());
                let downloaded_path = repo
                    .get("qwen2.5-0.5b-instruct-q4_k_m.gguf")
                    .context("Failed to download model from Hugging Face")?;
                std::fs::copy(&downloaded_path, &local_model)?;
            }

            local_model
        };

        // 2. Load the GGUF model with candle_core::quantized
        let mut file = std::fs::File::open(&model_path).context("Failed to open model file")?;
        let content = gguf_file::Content::read(&mut file).context("Failed to read GGUF content")?;

        // Use CPU for background daemon to save VRAM for other apps;
        // using quantized weights ensures this is still fast.
        let model = ModelWeights::from_gguf(content, &mut file, &Device::Cpu)
            .context("Failed to load quantized model weights. Ensure you are using a GGUF file with Q4_K_M or similar quantization.")?;

        // 3. Load Tokenizer
        let local_tokenizer = config.models_dir.join("qwen-tokenizer.json");
        let tokenizer_path = if local_tokenizer.exists() {
            tracing::info!("Found local tokenizer at {:?}", local_tokenizer);
            local_tokenizer
        } else {
            tracing::info!("Local tokenizer not found in models_dir, provisioning...");

            let qwen_tokenizer_url =
                "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";
            if let Err(e) = download_direct(qwen_tokenizer_url, &local_tokenizer).await {
                tracing::warn!(
                    "Direct download of Qwen tokenizer failed: {}. Falling back to HF Hub...",
                    e
                );
                let api = Api::new().context("Failed to create HF API client")?;
                let token_repo = api.model("Qwen/Qwen2.5-0.5B-Instruct".to_string());
                let downloaded_path = token_repo
                    .get("tokenizer.json")
                    .or_else(|_| token_repo.get("onnx/tokenizer.json"))
                    .context("Failed to download tokenizer from Hugging Face")?;
                std::fs::copy(&downloaded_path, &local_tokenizer)?;
            }

            local_tokenizer
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e)
        })?;

        Ok(Self {
            model: Some(Mutex::new(model)),
            tokenizer: Some(tokenizer),
            quantizer,
            cache: Arc::new(dashmap::DashMap::new()),
            semantic_cache: Arc::new(dashmap::DashMap::new()),
        })
    }

    /// Summarize a text chunk using the pre-loaded local LLM.
    pub async fn summarize_chunk(
        &self,
        text: &str,
        vector: Option<&[f32]>,
        config: Arc<Config>,
    ) -> Result<String> {
        if !config.feature_toggles.enable_local_llm {
            return Ok("LLM Disabled".to_string());
        }

        // 1. Exact Match Cache Check
        let hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            hex::encode(hasher.finalize())
        };

        if let Some(cached) = self.cache.get(&hash) {
            tracing::debug!("Exact cache hit for code chunk");
            return Ok(cached.clone());
        }

        self.summarize_chunk_inner(text, vector, config, hash).await
    }

    /// Optimized version that takes an already computed TurboCode (as bytes).
    #[allow(dead_code)]
    pub async fn summarize_chunk_with_code(
        &self,
        text: &str,
        code_bytes: Option<&[u8]>,
        config: Arc<Config>,
    ) -> Result<String> {
        if !config.feature_toggles.enable_local_llm {
            return Ok("LLM Disabled".to_string());
        }

        let hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            hex::encode(hasher.finalize())
        };

        if let Some(cached) = self.cache.get(&hash) {
            return Ok(cached.clone());
        }

        // Semantic cache check using provided code_bytes
        if let (Some(bytes), Some(q)) = (code_bytes, &self.quantizer) {
            if let Ok(query_code) = bincode::deserialize::<turbo_quant::TurboCode>(bytes) {
                // Decode the query code to get an approximate vector for comparison
                if let Ok(query_vec) = q.decode_approximate(&query_code) {
                    for entry in self.semantic_cache.iter() {
                        let (stored_code_bytes, summary) = entry.pair();
                        if let Ok(stored_code) =
                            bincode::deserialize::<turbo_quant::TurboCode>(stored_code_bytes)
                        {
                            // Compare the query vector against the stored code
                            if let Ok(sim) = q.inner_product_estimate(&stored_code, &query_vec) {
                                // Normalize by vector norms for cosine-like similarity
                                let query_norm: f32 =
                                    query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                                let stored_norm: f32 = stored_code
                                    .polar_code
                                    .radii
                                    .iter()
                                    .map(|r| r * r)
                                    .sum::<f32>()
                                    .sqrt();
                                let normalized_sim = if query_norm > 0.0 && stored_norm > 0.0 {
                                    sim / (query_norm * stored_norm)
                                } else {
                                    0.0
                                };

                                if normalized_sim > 0.98 {
                                    tracing::debug!(
                                        "Semantic cache hit (sim: {:.4})",
                                        normalized_sim
                                    );
                                    self.cache.insert(hash, summary.clone());
                                    return Ok(summary.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback to generation, then retain the precomputed code for future
        // semantic-cache hits.
        let summary = self
            .summarize_chunk_inner(text, None, config, hash.clone())
            .await?;
        if let Some(bytes) = code_bytes {
            self.semantic_cache.insert(bytes.to_vec(), summary.clone());
        }
        Ok(summary)
    }

    async fn summarize_chunk_inner(
        &self,
        text: &str,
        vector: Option<&[f32]>,
        _config: Arc<Config>,
        hash: String,
    ) -> Result<String> {
        // 2. Semantic Cache Check (using TurboQuant)
        let mut turbo_code = None;
        if let (Some(v), Some(q)) = (vector, &self.quantizer) {
            if let Ok(code) = q.encode(v) {
                // Check against existing codes in semantic cache
                for entry in self.semantic_cache.iter() {
                    let (stored_code_bytes, summary) = entry.pair();
                    if let Ok(stored_code) =
                        bincode::deserialize::<turbo_quant::TurboCode>(stored_code_bytes)
                    {
                        if let Ok(sim) = q.inner_product_estimate(&stored_code, v) {
                            if sim > 0.98 {
                                tracing::debug!("Semantic cache hit (sim: {:.4})", sim);
                                self.cache.insert(hash, summary.clone());
                                return Ok(summary.clone());
                            }
                        }
                    }
                }

                if let Ok(bytes) = bincode::serialize(&code) {
                    turbo_code = Some(bytes);
                }
            }
        }

        tracing::debug!("Summarizing code chunk (length: {})", text.len());
        let model_mutex = self.model.as_ref().context("Model not loaded")?;
        let tokenizer = self.tokenizer.as_ref().context("Tokenizer not loaded")?;

        // Aggressively truncate input to minimize prefill time on CPU
        // Most code summaries can be generated from first 800 chars
        let truncated_text = if text.len() > 800 { &text[..800] } else { text };

        // 4. Tokenize prompt - use shorter system prompt for speed
        let prompt = format!(
            "<|im_start|>system\nSummarize code concisely.<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
            truncated_text
        );
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let mut tokens = encoding.get_ids().to_vec();
        let mut generated_tokens = Vec::new();

        // 5. Optimized generation loop - reduce max tokens to 30 for speed
        // Most summaries are 10-20 tokens anyway
        let mut model = model_mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("Model mutex poisoned"))?;

        const MAX_TOKENS: usize = 30;
        const EOS_TOKEN: u32 = 151643;
        const NEWLINE_TOKEN: u32 = 151645;

        for i in 0..MAX_TOKENS {
            let context_size = if i == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len() - context_size;
            let input_ids = Tensor::new(&tokens[start_pos..], &Device::Cpu)?.unsqueeze(0)?;

            let logits = model.forward(&input_ids, start_pos)?;
            let logits = logits.squeeze(0)?; // [seq, vocab]

            // Get the logits for the last token in the sequence
            let last_token_logits = if logits.dims().len() > 1 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };

            let next_token = last_token_logits.argmax(0)?.to_scalar::<u32>()?;

            // Stop on EOS or newline (summaries should be single line)
            if next_token == EOS_TOKEN || next_token == NEWLINE_TOKEN {
                break;
            }

            tokens.push(next_token);
            generated_tokens.push(next_token);
        }

        let output = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?
            .trim()
            .to_string();

        // Save to cache
        self.cache.insert(hash, output.clone());
        if let Some(code_bytes) = turbo_code {
            self.semantic_cache.insert(code_bytes, output.clone());
        }

        Ok(output)
    }
}
