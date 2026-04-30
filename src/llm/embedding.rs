#![allow(dead_code)]
use crate::config::Config;
use crate::llm::models::get_model_registry;
use crate::llm::util::download_direct;
use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use tracing::info;
use turbo_quant::{TurboCode, TurboQuantizer};

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
    pub quantizer: Option<TurboQuantizer>,
}

impl Embedder {
    pub async fn new(config: &Config) -> Result<Self> {
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
                // Note: To use quantized models, ensure the URL points to a model_quantized.onnx file.
                download_direct(m_cfg.onnx_url, &local_onnx_path).await?;
                download_direct(m_cfg.tokenizer_url, &local_tokenizer_path).await?;
            } else {
                // Fallback to HF Hub for unknown models
                let api = Api::new().context("Failed to create HF API client")?;
                let repo = api.model(config.model_name.clone());

                if !local_onnx_path.exists() {
                    // Prioritize INT8 quantized models for lower RAM and higher speed
                    let model_path = repo
                        .get("onnx/model_quantized.onnx")
                        .or_else(|_| repo.get("onnx/model.onnx"))
                        .or_else(|_| repo.get("model.onnx"))
                        .context("Failed to download ONNX model from HF")?;
                    std::fs::copy(&model_path, &local_onnx_path)?;
                }

                if !local_tokenizer_path.exists() {
                    let tokenizer_path = repo
                        .get("tokenizer.json")
                        .or_else(|_| repo.get("onnx/tokenizer.json"))
                        .context("Failed to download tokenizer.json from HF")?;
                    std::fs::copy(&tokenizer_path, &local_tokenizer_path)?;
                }
            }
        }

        // Configure ORT session with optimal execution providers (EP)
        let builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create ORT session builder: {e}"))?;

        // Attempt to enable hardware acceleration
        #[cfg(target_os = "macos")]
        let builder = builder
            .with_execution_providers([ort::execution_providers::CoreMLExecutionProvider::default(
            )
            .build()])
            .map_err(|e| anyhow::anyhow!("Failed to enable CoreML EP: {e}"))?;

        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        let builder = builder
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build()
            ])
            .map_err(|e| anyhow::anyhow!("Failed to enable CUDA EP: {e}"))?;

        let session = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {e}"))?
            .commit_from_file(&local_onnx_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {e}"))?;

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
                    download_direct(cfg.onnx_url, &r_onnx).await?;
                    download_direct(cfg.tokenizer_url, &r_tok).await?;
                } else {
                    let api = Api::new().context("Failed to create HF API client for reranker")?;
                    let r_repo = api.model(config.reranker_model_name.clone());
                    let m_path = r_repo
                        .get("onnx/model_quantized.onnx")
                        .or_else(|_| r_repo.get("onnx/model.onnx"))
                        .or_else(|_| r_repo.get("model.onnx"))?;
                    let t_path = r_repo
                        .get("tokenizer.json")
                        .or_else(|_| r_repo.get("onnx/tokenizer.json"))?;
                    std::fs::copy(&m_path, &r_onnx)?;
                    std::fs::copy(&t_path, &r_tok)?;
                }
            }

            let sess = Session::builder()
                .map_err(|e| anyhow::anyhow!("Failed to create reranker session builder: {e}"))?
                .commit_from_file(r_onnx)
                .map_err(|e| anyhow::anyhow!("Failed to load reranker ONNX model: {e}"))?;
            let tok = build_tokenizer(r_tok)?;
            rerank_session = Some(Mutex::new(sess));
            rerank_tokenizer = Some(tok);
        }

        // Initialize TurboQuantizer if dimension is suitable (must be even)
        let quantizer = if config.dimension % 2 == 0 {
            let projections = config.dimension / 4;
            TurboQuantizer::new(config.dimension, 8, projections, 42).ok()
        } else {
            None
        };

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_name: config.model_name.clone(),
            dimension: config.dimension,
            matryoshka_dim,
            rerank_session,
            rerank_tokenizer,
            quantizer,
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

    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let q = self
            .quantizer
            .as_ref()
            .context("TurboQuantizer not initialized (dimension must be even)")?;
        let code = q
            .encode(vector)
            .map_err(|e| anyhow::anyhow!("Quantization failed: {e}"))?;

        bincode::serialize(&code).context("Failed to serialize TurboCode")
    }

    pub fn estimate_similarity(&self, code_bytes: &[u8], query: &[f32]) -> Result<f32> {
        let q = self
            .quantizer
            .as_ref()
            .context("TurboQuantizer not initialized")?;

        let code: TurboCode =
            bincode::deserialize(code_bytes).context("Failed to deserialize TurboCode")?;

        q.inner_product_estimate(&code, query)
            .map_err(|e| anyhow::anyhow!("Similarity estimation failed: {e}"))
    }

    /// Embed a batch of texts in a single ONNX forward pass.
    /// Returns one embedding vector per input string.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let batch_size = texts.len();

        // Tokenize each text and track the actual max sequence length in this batch.
        let mut encodings = Vec::with_capacity(batch_size);
        let mut max_len: usize = 0;
        for text in texts {
            let adapted = apply_adapter(text, &self.model_name, false);
            let enc = self
                .tokenizer
                .encode(adapted.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
            let seq_len = enc.get_ids().len().min(MAX_SEQ_LEN);
            if seq_len > max_len {
                max_len = seq_len;
            }
            encodings.push(enc);
        }

        // Build flat padded arrays using the dynamic max_len (not MAX_SEQ_LEN).
        let mut all_ids = vec![0i64; batch_size * max_len];
        let mut all_mask = vec![0i64; batch_size * max_len];
        let mut all_type_ids = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let type_ids = enc.get_type_ids();
            let seq_len = ids.len().min(max_len);
            let offset = i * max_len;
            for j in 0..seq_len {
                all_ids[offset + j] = ids[j] as i64;
                all_mask[offset + j] = mask[j] as i64;
                all_type_ids[offset + j] = type_ids[j] as i64;
            }
        }

        let input_ids = Array2::from_shape_vec((batch_size, max_len), all_ids)?;
        let attention_mask = Array2::from_shape_vec((batch_size, max_len), all_mask)?;
        let token_type_ids = Array2::from_shape_vec((batch_size, max_len), all_type_ids)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Embedder mutex poisoned"))?;

        let mut inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids).map_err(|e| anyhow::anyhow!("Failed to create input_ids: {e}"))?,
            "attention_mask" => Value::from_array(attention_mask).map_err(|e| anyhow::anyhow!("Failed to create attention_mask: {e}"))?,
        ];
        if session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids")
        {
            inputs.push((
                "token_type_ids".into(),
                Value::from_array(token_type_ids)
                    .map_err(|e| anyhow::anyhow!("Failed to create token_type_ids: {e}"))?
                    .into(),
            ));
        }

        let outputs = session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("Session run failed: {e}"))?;
        let output_value = outputs
            .get("last_hidden_state")
            .or_else(|| outputs.get("output"))
            .unwrap_or(&outputs[0]);
        let output_array = output_value
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output array: {e}"))?;

        let target_dim = self.matryoshka_dim.unwrap_or(self.dimension);
        let mut result = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut emb = output_array.slice(ndarray::s![i, 0, ..target_dim]).to_vec();
            normalize(&mut emb);
            result.push(emb);
        }
        Ok(result)
    }

    pub fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<f32>> {
        let session_mutex = self
            .rerank_session
            .as_ref()
            .context("Reranker not loaded")?;
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
            // Truncate or pad each sequence to exactly MAX_SEQ_LEN so the
            // flat Vec has a consistent stride for Array2::from_shape_vec.
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let seq_len = ids.len().min(MAX_SEQ_LEN);
            for j in 0..MAX_SEQ_LEN {
                all_ids.push(if j < seq_len { ids[j] as i64 } else { 0 });
                all_mask.push(if j < seq_len { mask[j] as i64 } else { 0 });
            }
        }

        let input_ids = Array2::from_shape_vec((batch_size, MAX_SEQ_LEN), all_ids)?;
        let attention_mask = Array2::from_shape_vec((batch_size, MAX_SEQ_LEN), all_mask)?;

        let mut session = session_mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("Reranker mutex poisoned"))?;

        let mut inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids).map_err(|e| anyhow::anyhow!("Failed to create input_ids: {e}"))?,
            "attention_mask" => Value::from_array(attention_mask).map_err(|e| anyhow::anyhow!("Failed to create attention_mask: {e}"))?,
        ];

        // Handle models that don't support token_type_ids or have different names.
        // MS-Marco and BGE-Reranker usually just need the first two.
        if session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids")
        {
            let type_ids: Vec<i64> = vec![0; batch_size * MAX_SEQ_LEN];
            let token_type_ids = Array2::from_shape_vec((batch_size, MAX_SEQ_LEN), type_ids)?;
            inputs.push((
                "token_type_ids".into(),
                Value::from_array(token_type_ids)
                    .map_err(|e| anyhow::anyhow!("Failed to create token_type_ids: {e}"))?
                    .into(),
            ));
        }

        let outputs = session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("Reranker session run failed: {e}"))?;

        let arr = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract reranker output: {e}"))?;
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
            "input_ids" => Value::from_array(input_ids).map_err(|e| anyhow::anyhow!("Failed to create input_ids: {e}"))?,
            "attention_mask" => Value::from_array(attention_mask).map_err(|e| anyhow::anyhow!("Failed to create attention_mask: {e}"))?,
        ];

        if session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids")
        {
            inputs.push((
                "token_type_ids".into(),
                Value::from_array(token_type_ids)
                    .map_err(|e| anyhow::anyhow!("Failed to create token_type_ids: {e}"))?
                    .into(),
            ));
        }

        let outputs = session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("Session run failed: {e}"))?;

        let output_value = outputs
            .get("last_hidden_state")
            .or_else(|| outputs.get("output"))
            .unwrap_or(&outputs[0]);

        let output_array = output_value
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output array: {e}"))?;

        // CLS pooling: token 0 of the sequence.
        let full_dim = self.dimension;
        let target_dim = self.matryoshka_dim.unwrap_or(full_dim);

        let mut embedding = output_array.slice(ndarray::s![0, 0, ..target_dim]).to_vec();

        normalize(&mut embedding);
        Ok(embedding)
    }
}

fn build_tokenizer(path: std::path::PathBuf) -> Result<Tokenizer> {
    let mut tok =
        Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

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
