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
    pub rerank_session: Option<Mutex<Session>>,
    pub rerank_tokenizer: Option<Tokenizer>,
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

        // --- Provision Reranker (Optional) ---
        let mut rerank_session = None;
        let mut rerank_tokenizer = None;

        if !config.reranker_model_name.is_empty() {
            info!(model = %config.reranker_model_name, "Provisioning reranker model...");
            let r_repo = api.model(config.reranker_model_name.clone());
            if let Ok(r_model) = r_repo.get("model.onnx").or_else(|_| r_repo.get("onnx/model.onnx")) {
                if let Ok(r_sess) = Session::builder()?.commit_from_file(r_model) {
                    if let Ok(r_tok_path) = r_repo.get("tokenizer.json").or_else(|_| r_repo.get("onnx/tokenizer.json")) {
                        if let Ok(mut r_tok) = Tokenizer::from_file(r_tok_path) {
                            let _ = r_tok.with_truncation(Some(tokenizers::TruncationParams {
                                max_length: 512,
                                ..Default::default()
                            }));
                            let _ = r_tok.with_padding(Some(tokenizers::PaddingParams {
                                strategy: tokenizers::PaddingStrategy::Fixed(512),
                                ..Default::default()
                            }));
                            rerank_session = Some(Mutex::new(r_sess));
                            rerank_tokenizer = Some(r_tok);
                        }
                    }
                }
            }
        }
        
        Ok(Self { 
            session: Mutex::new(session), 
            tokenizer,
            rerank_session,
            rerank_tokenizer,
        })
    }

    /// Embed a single text chunk.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

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

        let output_value = outputs.get("last_hidden_state").or_else(|| outputs.get("output")).unwrap_or(&outputs[0]);
        let output_array = output_value.try_extract_array::<f32>()?;
        let embedding = output_array.slice(ndarray::s![0, 0, ..]).to_vec();

        Ok(embedding)
    }

    /// Rerank a set of documents against a query using batch processing for better CPU performance.
    pub fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<f32>> {
        let session_mutex = self.rerank_session.as_ref().context("Reranker not loaded")?;
        let tokenizer = self.rerank_tokenizer.as_ref().context("Reranker tokenizer not loaded")?;
        
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut session = session_mutex.lock().map_err(|_| anyhow::anyhow!("Reranker mutex poisoned"))?;
        let batch_size = documents.len();
        
        // Pre-allocate vectors for batching
        let mut all_ids = Vec::with_capacity(batch_size * 512);
        let mut all_mask = Vec::with_capacity(batch_size * 512);

        for doc in documents {
            let encoding = tokenizer.encode((query.to_string(), doc), true)
                .map_err(|e| anyhow::anyhow!("Rerank tokenization failed: {}", e))?;
            
            all_ids.extend(encoding.get_ids().iter().map(|&id| id as i64));
            all_mask.extend(encoding.get_attention_mask().iter().map(|&m| m as i64));
        }

        let input_ids = Array2::from_shape_vec((batch_size, 512), all_ids)?;
        let attention_mask = Array2::from_shape_vec((batch_size, 512), all_mask)?;

        let outputs = session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids)?,
            "attention_mask" => Value::from_array(attention_mask)?,
        ])?;

        let output_value = &outputs[0];
        let output_array = output_value.try_extract_array::<f32>()?;
        
        // Extract scores for the first token (usually the classification token)
        let mut scores = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            scores.push(output_array[[i, 0]]);
        }

        Ok(scores)
    }
}
