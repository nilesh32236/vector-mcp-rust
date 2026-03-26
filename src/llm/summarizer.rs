use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use hf_hub::api::sync::Api;
use crate::config::Config;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct Summarizer {}

impl Summarizer {
    /// Create a new Summarizer.
    pub fn new(_config: &Config) -> Result<Self> {
        Ok(Self {})
    }

    /// Summarize a text chunk using a standalone LLM.
    /// Follows the rule: If config.features.enable_local_llm is false, return "LLM Disabled".
    pub async fn summarize_chunk(&self, text: &str, config: Arc<Config>) -> Result<String> {
        if !config.feature_toggles.enable_local_llm {
            return Ok("LLM Disabled".to_string());
        }

        // 1. Download/Cache model from Hugging Face
        // Repository and file requested: "Qwen/Qwen2.5-0.5B-Instruct-GGUF" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        let api = Api::new().context("Failed to create HF API client")?;
        let repo = api.model("Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string());
        let model_path = repo.get("qwen2.5-0.5b-instruct-q4_k_m.gguf")
            .context("Failed to download/access model from Hugging Face")?;

        // 2. Load the GGUF model
        let mut file = std::fs::File::open(&model_path).context("Failed to open model file")?;
        let content = gguf_file::Content::read(&mut file).context("Failed to read GGUF content")?;
        
        // Load into CPU for memory safety and maximum compatibility
        let mut model = ModelWeights::from_gguf(content, &mut file, &Device::Cpu)
            .context("Failed to load quantized model weights")?;

        // 3. Pre-load Tokenizer for the LLM
        // We'll use the main instruct repo for the tokenizer config
        let token_repo = api.model("Qwen/Qwen2.5-0.5B-Instruct".to_string());
        let tokenizer_path = token_repo.get("tokenizer.json")
            .context("Failed to download tokenizer from Hugging Face")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // 4. Tokenize prompt
        let prompt = format!("<|im_start|>system\nYou are a helpful assistant that summarizes code chunks.<|im_end|>\n<|im_start|>user\nSummarize the following code in one sentence: {}\n<|im_end|>\n<|im_start|>assistant\n", text);
        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        let mut tokens = encoding.get_ids().to_vec();
        let mut generated_tokens = Vec::new();
        
        // 5. Basic generation loop (max 100 tokens)
        for i in 0..100 {
            let context_size = if i == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len() - context_size;
            let input_ids = Tensor::new(&tokens[start_pos..], &Device::Cpu)?.unsqueeze(0)?;
            
            let logits = model.forward(&input_ids, start_pos)?;
            let logits = logits.squeeze(0)?;
            
            // Greedy sampling for speed and predictability
            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
            
            // Check for EOS (Qwen2 EOS is usually 151643 or similar)
            if next_token == 151643 || next_token == 151645 {
                break;
            }
            
            tokens.push(next_token);
            generated_tokens.push(next_token);
        }

        let output = tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(output.trim().to_string())
    }
}
