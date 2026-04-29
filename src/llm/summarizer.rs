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
    cache: Arc<dashmap::DashMap<String, String>>, // Semantic cache: fingerprint -> summary
}

impl Summarizer {
    /// Create a new Summarizer and load the model if local LLM is enabled.
    pub async fn new(config: &Config) -> Result<Self> {
        if !config.feature_toggles.enable_local_llm {
            return Ok(Self {
                model: None,
                tokenizer: None,
            });
        }

        // 1. Download/Cache model from Hugging Face
        let local_model = config.models_dir.join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
        let model_path = if local_model.exists() {
            tracing::info!("Found local model at {:?}", local_model);
            local_model
        } else {
            tracing::info!("Local model not found in models_dir, provisioning...");
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

        // 2. Load the GGUF model
        let mut file = std::fs::File::open(&model_path).context("Failed to open model file")?;
        let content = gguf_file::Content::read(&mut file).context("Failed to read GGUF content")?;

        let model = ModelWeights::from_gguf(content, &mut file, &Device::Cpu)
            .context("Failed to load quantized model weights")?;

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

        let quantizer = turbo_quant::TurboQuantizer::new(384, 8, 96, 42).ok();

        Ok(Self {
            model: Some(Mutex::new(model)),
            tokenizer: Some(tokenizer),
            quantizer,
            cache: Arc::new(dashmap::DashMap::new()),
        })
    }

    /// Summarize a text chunk using the pre-loaded local LLM.
    pub async fn summarize_chunk(&self, text: &str, config: Arc<Config>) -> Result<String> {
        if !config.feature_toggles.enable_local_llm {
            return Ok("LLM Disabled".to_string());
        }

        // 1. Semantic Cache Check (using a simple hash for now, could be TurboQuant distance)
        let hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            hex::encode(hasher.finalize())
        };

        if let Some(cached) = self.cache.get(&hash) {
            tracing::debug!("Semantic cache hit for code chunk");
            return Ok(cached.clone());
        }

        tracing::info!("Summarizing code chunk (length: {})", text.len());
        let model_mutex = self.model.as_ref().context("Model not loaded")?;
        let tokenizer = self.tokenizer.as_ref().context("Tokenizer not loaded")?;

        // Truncate input to avoid long prefill times on CPU
        let truncated_text = if text.len() > 2000 {
            &text[..2000]
        } else {
            text
        };

        // 4. Tokenize prompt
        let prompt = format!(
            "<|im_start|>system\nYou are a helpful assistant that summarizes code chunks.<|im_end|>\n<|im_start|>user\nSummarize the following code in one sentence: {}\n<|im_end|>\n<|im_start|>assistant\n",
            truncated_text
        );
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let mut tokens = encoding.get_ids().to_vec();
        let mut generated_tokens = Vec::new();

        // 5. Basic generation loop (max 50 tokens for speed)
        let mut model = model_mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("Model mutex poisoned"))?;

        for i in 0..50 {
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

            if next_token == 151643 || next_token == 151645 {
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

        Ok(output)
    }
}
