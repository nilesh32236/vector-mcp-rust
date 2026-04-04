#![allow(dead_code)]
use crate::config::Config;
use crate::llm::models::{get_model_registry, ModelConfig};
use crate::llm::util::download_direct;
use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use tracing::{info, warn};

const MAX_SEQ_LEN: usize = 512;

// ---------------------------------------------------------------------------
// Model adapter — mirrors Go's ModelAdapter / NomicAdapter
// ---------------------------------------------------------------------------

fn apply_adapter(text: &str, model_name: &str, is_query: bool) -> String {
    if model_name.to_lowercase().contains("nomic") {
        let prefix = if is_query {
            "search_query: "
        } else {
            "search_document: "
        };
        if text.starts_with(prefix) {
            text.to_string()
        } else {
            format!("{prefix}{text}")
        }
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------
// L2 normalisation
// ---------------------------------------------------------------------------

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

pub struct Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    model_name: String,
    /// Full output dimension from the ONNX model.
    dimension: usize,
    /// Optional Matryoshka truncation target (None = no truncation).
    matryoshka_dim: Option<usize>,
    pub rerank_session: Option<Mutex<Session>>,
    pub rerank_tokenizer: Option<Tokenizer>,
}

impl Embedder {
    pub fn new(config: &Config) -> Result<Self> {
        info!(model = %config.model_name, "Provisioning embedder model...");
        
        let registry = get_model_registry();
        let model_cfg = registry.get(config.model_name.as_str());

        let model_filename = config.model_name.replace('/', "_");
        let local_model_dir = config.models_dir.join(&model_filename);
        let local_onnx_path = local_model_dir.join("model.onnx");
        let local_tokenizer_path = local_model_dir.join("tokenizer.json");

        if !local_onnx_path.exists() || !local_tokenizer_path.exists() {
            info!("Model files not found locally, provisioning...");
            std::fs::create_dir_all(&local_model_dir)?;
            
            if let Some(m_cfg) = model_cfg {
                // Use Direct Download for known registry models
                download_direct(m_cfg.onnx_url, &local_onnx_path)?;
                download_direct(m_cfg.tokenizer_url, &local_tokenizer_path)?;
            } else {
                // Fallback to HF Hub for unknown models
                let api = Api::new().context("Failed to create HF API client")?;
                let repo = api.model(config.model_name.clone());

                if !local_onnx_path.exists() {
                    let model_path = repo.get("onnx/model_quantized.onnx")
                        .or_else(|_| repo.get("onnx/model.onnx"))
                        .or_else(|_| repo.get("model.onnx"))
                        .context("Failed to download ONNX model from HF")?;
                    std::fs::copy(&model_path, &local_onnx_path)?;
                }

                if !local_tokenizer_path.exists() {
                    let tokenizer_path = repo.get("tokenizer.json")
                        .or_else(|_| repo.get("onnx/tokenizer.json"))
                        .context("Failed to download tokenizer.json from HF")?;
                    std::fs::copy(&tokenizer_path, &local_tokenizer_path)?;
                }
            }
        }

        let session = Session::builder()?
            .commit_from_file(&local_onnx_path)
            .context("Failed to load ONNX model")?;

        let tokenizer = build_tokenizer(local_tokenizer_path)?;

        // Resolve Matryoshka dim
        let matryoshka_dim = std::env::var("MATRYOSHKA_DIM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&d| d > 0 && d < config.dimension);

        // --- Optional reranker ---
        let mut rerank_session = None;
        let mut rerank_tokenizer = None;
        if !config.reranker_model_name.is_empty() {
            info!(model = %config.reranker_model_name, "Provisioning reranker...");
            let r_cfg = registry.get(config.reranker_model_name.as_str());
            
            let r_filename = config.reranker_model_name.replace('/', "_");
            let r_dir = config.models_dir.join(&r_filename);
            let r_onnx = r_dir.join("model.onnx");
            let r_tok = r_dir.join("tokenizer.json");

            if !r_onnx.exists() || !r_tok.exists() {
                std::fs::create_dir_all(&r_dir)?;
                if let Some(cfg) = r_cfg {
                    download_direct(cfg.onnx_url, &r_onnx)?;
                    download_direct(cfg.tokenizer_url, &r_tok)?;
                } else {
                    let api = Api::new().context("Failed to create HF API client for reranker")?;
                    let r_repo = api.model(config.reranker_model_name.clone());
                    let m_path = r_repo.get("onnx/model_quantized.onnx")
                        .or_else(|_| r_repo.get("onnx/model.onnx"))
                        .or_else(|_| r_repo.get("model.onnx"))?;
                    let t_path = r_repo.get("tokenizer.json")
                        .or_else(|_| r_repo.get("onnx/tokenizer.json"))?;
                    std::fs::copy(&m_path, &r_onnx)?;
                    std::fs::copy(&t_path, &r_tok)?;
                }
            }

            let sess = Session::builder()?.commit_from_file(r_onnx)?;
            let tok = build_tokenizer(r_tok)?;
            rerank_session = Some(Mutex::new(sess));
            rerank_tokenizer = Some(tok);
        }

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_name: config.model_name.clone(),
            dimension: config.dimension,
            matryoshka_dim,
            rerank_session,
            rerank_tokenizer,
        })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_inner(text, false)
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_inner(text, true)
    }

    pub fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<f32>> {
        let session_mutex = self.rerank_session.as_ref().context("Reranker not loaded")?;
        let tokenizer = self
            .rerank_tokenizer
            .as_ref()
            .context("Reranker tokenizer not loaded")?;

        if documents.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = documents.len();
        let mut all_ids = Vec::with_capacity(batch_size * MAX_SEQ_LEN);
        let mut all_mask = Vec::with_capacity(batch_size * MAX_SEQ_LEN);

        for doc in &documents {
            let enc = tokenizer
                .encode((query.to_string(), doc.clone()), true)
                .map_err(|e| anyhow::anyhow!("Rerank tokenization failed: {e}"))?;
            all_ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            all_mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
        }

        let input_ids = Array2::from_shape_vec((batch_size, MAX_SEQ_LEN), all_ids)?;
        let attention_mask = Array2::from_shape_vec((batch_size, MAX_SEQ_LEN), all_mask)?;

        let mut session = session_mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("Reranker mutex poisoned"))?;

        let mut inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids)?,
            "attention_mask" => Value::from_array(attention_mask)?,
        ];

        // Handle models that don't support token_type_ids or have different names.
        // MS-Marco and BGE-Reranker usually just need the first two.
        if session.inputs().iter().any(|i| i.name() == "token_type_ids") {
            let type_ids: Vec<i64> = vec![0; batch_size * MAX_SEQ_LEN];
            let token_type_ids = Array2::from_shape_vec((batch_size, MAX_SEQ_LEN), type_ids)?;
            inputs.push(("token_type_ids".into(), Value::from_array(token_type_ids)?.into()));
        }

        let outputs = session.run(inputs)?;

        let arr = outputs[0].try_extract_array::<f32>()?;
        Ok((0..batch_size).map(|i| arr[[i, 0]]).collect())
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn embed_inner(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        let text = apply_adapter(text, &self.model_name, is_query);

        let enc = self
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        let type_ids: Vec<i64> = enc.get_type_ids().iter().map(|&x| x as i64).collect();

        let input_ids = Array2::from_shape_vec((1, MAX_SEQ_LEN), ids)?;
        let attention_mask = Array2::from_shape_vec((1, MAX_SEQ_LEN), mask)?;
        let token_type_ids = Array2::from_shape_vec((1, MAX_SEQ_LEN), type_ids)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Embedder mutex poisoned"))?;

        // Dynamically build inputs based on what the model expects.
        // Some models (like BGE-M3) don't want token_type_ids.
        let mut inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids)?,
            "attention_mask" => Value::from_array(attention_mask)?,
        ];

        if session.inputs().iter().any(|i| i.name() == "token_type_ids") {
            inputs.push(("token_type_ids".into(), Value::from_array(token_type_ids)?.into()));
        }

        let outputs = session.run(inputs)?;

        let output_value = outputs
            .get("last_hidden_state")
            .or_else(|| outputs.get("output"))
            .unwrap_or(&outputs[0]);

        let output_array = output_value.try_extract_array::<f32>()?;

        // CLS pooling: token 0 of the sequence.
        let full_dim = self.dimension;
        let target_dim = self.matryoshka_dim.unwrap_or(full_dim);

        let mut embedding = output_array
            .slice(ndarray::s![0, 0, ..target_dim])
            .to_vec();

        normalize(&mut embedding);
        Ok(embedding)
    }
}

fn build_tokenizer(path: std::path::PathBuf) -> Result<Tokenizer> {
    let mut tok = Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    let _ = tok.with_truncation(Some(tokenizers::TruncationParams {
        max_length: MAX_SEQ_LEN,
        ..Default::default()
    }));
    let _ = tok.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(MAX_SEQ_LEN),
        ..Default::default()
    }));

    Ok(tok)
}
