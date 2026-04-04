pub mod chunker;
pub mod scanner;
pub mod watcher;

use anyhow::{Context, Result};
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};

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
            if name == "node_modules" || name == ".git" || name == "target" || name == ".next" || name == ".data" {
                continue;
            }

            item_count += 1;
            out.push_str(&format!("{}├── {}\n", "│   ".repeat(depth - 1), name));
        }
    }

    Ok(out)
}

/// Indexes a single file, replacing any existing vectors for that path.
pub async fn index_file(
    path: &str,
    config: Arc<Config>,
    store: Arc<Store>,
    embedder: Arc<Embedder>,
    summarizer: Arc<Summarizer>,
) -> Result<()> {
    let raw_bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("Reading file for indexing: {path}"))?;

    let current_hash = compute_hash(&raw_bytes);

    // Check if the file already exists in the DB and has the same hash.
    if let Ok(existing_records) = store.get_records_by_path(path).await
        && !existing_records.is_empty() {
            let existing_hash = existing_records[0].content_hash();
            if existing_hash == current_hash {
                tracing::debug!("Skipping indexing for {}: content hash unchanged", path);
                return Ok(());
            }
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

    let chunks = if extension == ".pdf" {
        chunker::parse_pdf(&raw_bytes, path)?
    } else {
        let content = String::from_utf8_lossy(&raw_bytes);
        chunker::parse_file(&content, path, &extension)?
    };
    if chunks.is_empty() {
        return Ok(());
    }

    let mut records = Vec::new();
    let updated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs()
        .to_string();

    for (i, chunk) in chunks.into_iter().enumerate() {
        let vector = embedder.embed_text(&chunk.content)?;

        // Optional: Generate AI summary if local LLM is enabled.
        // We only do this for function/class chunks to save time, or if requested.
        let ai_summary: String =
            if config.feature_toggles.enable_local_llm && chunk.function_score > 0.5 {
                match summarizer
                    .summarize_chunk(&chunk.content, Arc::clone(&config))
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("Summary failed for {}: {}", path, e);
                        "Summary failed".to_string()
                    }
                }
            } else {
                "No AI summary generated".to_string()
            };

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
        });
    }

    store.delete_by_path(path).await?;
    store.upsert_records(records).await?;

    Ok(())
}

fn compute_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    hex::encode(result)
}
