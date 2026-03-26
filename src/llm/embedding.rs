use std::sync::Mutex;
use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;
use hf_hub::api::sync::Api;
use crate::config::Config;
use tracing::info;

pub struct Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl Embedder {
    /// Create a new Embedder session, auto-provisioning from Hugging Face if needed.
    pub fn new(config: &Config) -> Result<Self> {
        info!(model = %config.model_name, "Provisioning embedder model...");
        let api = Api::new().context("Failed to create HF API client")?;
        let repo = api.model(config.model_name.clone());
        
        // Try the common ONNX locations
        info!(model = %config.model_name, "Downloading model.onnx...");
        let model_path = repo.get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .context("Failed to download/access ONNX model from Hugging Face")?;
            
        info!(model = %config.model_name, "Downloading tokenizer.json...");
        let tokenizer_path = repo.get("tokenizer.json")
            .or_else(|_| repo.get("onnx/tokenizer.json"))
            .context("Failed to download/access tokenizer.json from Hugging Face")?;

        info!(model = %config.model_name, "Initialising ORT session...");
        let session = Session::builder()?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model into session")?;
            
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        // Enforce truncation to 512 tokens for stable inference
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            ..Default::default()
        }));

        // Enforce padding to 512 tokens for fixed-size tensor shapes
        let _ = tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(512),
            ..Default::default()
        }));
        
        Ok(Self { 
            session: Mutex::new(session), 
            tokenizer 
        })
    }

    /// Embed a single text chunk.
    /// Uses internal Mutex to allow sharing across threads with &self.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        // Shape: [batch_size, seq_len] -> [1, 512]
        let input_ids = Array2::from_shape_vec((1, 512), ids)?;
        let attention_mask = Array2::from_shape_vec((1, 512), mask)?;
        let token_type_ids = Array2::from_shape_vec((1, 512), type_ids)?;

        let mut session = self.session.lock()
            .map_err(|_| anyhow::anyhow!("Embedder mutex poisoned"))?;

        let outputs = session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids)?,
            "attention_mask" => Value::from_array(attention_mask)?,
            "token_type_ids" => Value::from_array(token_type_ids)?,
        ])?;

        let output_value = if let Some(val) = outputs.get("last_hidden_state") {
            val
        } else {
            &outputs[0]
        };

        let output_array = output_value.try_extract_array::<f32>()?;
        let embedding = output_array.slice(ndarray::s![0, 0, ..]).to_vec();

        Ok(embedding)
    }
}
