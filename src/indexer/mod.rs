pub mod chunker;
pub mod scanner;
pub mod watcher;

use anyhow::{Context, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::Config;
use crate::db::{Record, Store};
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;

pub fn get_directory_tree(path: &std::path::Path, max_depth: usize) -> Result<String> {
    let mut out = String::new();
    let mut item_count = 0;
    let max_items = 1000;

    let walker = ignore::WalkBuilder::new(path)
        .standard_filters(true)
        .add_custom_ignore_filename(".vector-ignore")
        .max_depth(Some(max_depth))
        .hidden(true)
        .build();

    for result in walker {
        if item_count >= max_items {
            break;
        }
        if let Ok(entry) = result {
            if entry.depth() == 0 {
                continue;
            }
            let name = entry.file_name().to_string_lossy();
            let depth = entry.depth();

            // Skip common noise
            if name == "node_modules"
                || name == ".git"
                || name == "target"
                || name == ".next"
                || name == ".data"
            {
                continue;
            }

            item_count += 1;
            out.push_str(&format!("{}├── {}\n", "│   ".repeat(depth - 1), name));
        }
    }

    Ok(out)
}

/// Indexes a single file: returns the records to be inserted (does NOT write to DB).
/// The caller is responsible for batching and flushing to the store.
pub async fn index_file(
    path: &str,
    config: Arc<Config>,
    store: Arc<Store>,
    embedder: Arc<Embedder>,
    _summarizer: Arc<Summarizer>,
) -> Result<Vec<Record>> {
    let raw_bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("Reading file for indexing: {path}"))?;

    let current_hash = compute_hash(&raw_bytes);

    // Skip if content unchanged.
    if let Ok(existing_records) = store.get_records_by_path(path).await
        && !existing_records.is_empty()
        && existing_records[0].content_hash() == current_hash
    {
        tracing::debug!("Skipping indexing for {}: content hash unchanged", path);
        return Ok(vec![]);
    }

    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            if e.starts_with('.') {
                e.to_string()
            } else {
                format!(".{}", e)
            }
        })
        .unwrap_or_default();

    // Offload CPU-bound parsing to a blocking thread so the async runtime stays free.
    let path_owned = path.to_string();
    let ext_owned = extension.clone();
    let chunks = tokio::task::spawn_blocking(move || -> Result<Vec<chunker::Chunk>> {
        if ext_owned == ".pdf" {
            chunker::parse_pdf(&raw_bytes, &path_owned)
        } else {
            let content = String::from_utf8_lossy(&raw_bytes).into_owned();
            chunker::parse_file(&content, &path_owned, &ext_owned)
        }
    })
    .await??;

    if chunks.is_empty() {
        return Ok(vec![]);
    }

    let updated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs()
        .to_string();

    // Batch-embed all chunks in one ONNX forward pass.
    const EMBED_BATCH_SIZE: usize = 32;
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
    for batch in chunk_texts.chunks(EMBED_BATCH_SIZE) {
        let batch = batch.to_vec();
        let embedder = Arc::clone(&embedder);
        let vecs = tokio::task::spawn_blocking(move || embedder.embed_batch(&batch)).await??;
        all_vectors.extend(vecs);
    }
    let quantized_codes: Vec<Option<Vec<u8>>> = all_vectors
        .iter()
        .map(|vector| embedder.quantize(vector).ok())
        .collect();

    // Summaries are intentionally deferred to the scanner's background worker.
    // Running local LLM generation inline makes initial indexing and API
    // readiness compete for the same CPU budget.
    let summaries = vec![String::new(); chunks.len()];

    let mut records = Vec::with_capacity(chunks.len());
    for (i, (((chunk, vector), quantized_code), ai_summary)) in chunks
        .into_iter()
        .zip(all_vectors)
        .zip(quantized_codes)
        .zip(summaries)
        .enumerate()
    {
        let metadata = json!({
            "path": path,
            "project_id": config.project_root.read().unwrap().clone(),
            "type": chunk.node_type,
            "updated_at": updated_at,
            "content_hash": current_hash,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbols": chunk.symbols,
            "calls": chunk.calls,
            "relationships": chunk.relationships,
            "function_score": chunk.function_score,
            "summary": ai_summary,
        });

        records.push(Record {
            id: format!(
                "{}-{}-{}",
                config.project_root.read().unwrap().clone(),
                path,
                i
            ),
            content: chunk.content,
            vector,
            metadata: metadata.to_string(),
            quantized_code,
        });
    }

    Ok(records)
}

fn compute_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    hex::encode(result)
}
