pub mod chunker;
pub mod scanner;
pub mod watcher;

use anyhow::{Context, Result};
use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};

use crate::config::Config;
use crate::db::{Record, Store};
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;

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
    if let Ok(existing_records) = store.get_records_by_path(path).await {
        if !existing_records.is_empty() {
            let existing_hash = existing_records[0].content_hash();
            if existing_hash == current_hash {
                tracing::debug!("Skipping indexing for {}: content hash unchanged", path);
                return Ok(());
            }
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

    let batch = records_to_batch(records, config.dimension)?;
    store.delete_by_path(path).await?;
    store.code_vectors.add(batch).execute().await?;

    Ok(())
}

fn records_to_batch(records: Vec<Record>, dimension: usize) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension as i32,
            ),
            false,
        ),
        Field::new("metadata", DataType::Utf8, true),
    ]));

    let ids = StringArray::from(records.iter().map(|r| r.id.as_str()).collect::<Vec<_>>());
    let contents = StringArray::from(
        records
            .iter()
            .map(|r| r.content.as_str())
            .collect::<Vec<_>>(),
    );
    let metadatas = StringArray::from(
        records
            .iter()
            .map(|r| r.metadata.as_str())
            .collect::<Vec<_>>(),
    );

    let mut flattened_vectors = Vec::new();
    for r in &records {
        flattened_vectors.extend_from_slice(&r.vector);
    }

    let vector_values = Float32Array::from(flattened_vectors);
    let vectors = FixedSizeListArray::try_new_from_values(vector_values, dimension as i32)?;

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(contents),
            Arc::new(vectors),
            Arc::new(metadatas),
        ],
    )
    .context("Creating RecordBatch for indexing")
}

fn compute_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    hex::encode(result)
}
