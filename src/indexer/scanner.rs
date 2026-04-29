use anyhow::Result;
use futures::StreamExt;
use ignore::{DirEntry, WalkBuilder};
use std::sync::Arc;
use tracing::{info, warn};

use crate::config::Config;
use crate::db::Store;
use crate::indexer;
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;

/// Strongly-typed indexing progress state.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProgressState {
    pub status: String,
    pub current_file: String,
    pub indexed_files: u64,
    pub total_files: u64,
    pub errors: Vec<String>,
}

/// Progress update sent to callers during a scan.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanProgress {
    pub progress_token: String,
    pub progress: u64,
    pub total: u64,
    pub current_file: String,
    pub status: &'static str,
}

const BATCH_INSERT_THRESHOLD: usize = 500;

/// Scans the entire project root and indexes all relevant source files.
/// Respects `.gitignore` and skip patterns via the `ignore` crate.
///
/// Uses a dedicated blocking thread to walk the directory (no upfront Vec
/// allocation) and streams paths to async workers via an mpsc channel.
/// Records are accumulated and flushed to LanceDB in batches.
pub async fn scan_project(
    config: Arc<Config>,
    store: Arc<Store>,
    embedder: Arc<Embedder>,
    summarizer: Arc<Summarizer>,
    progress: Arc<std::sync::RwLock<ProgressState>>,
    progress_tx: Option<tokio::sync::mpsc::Sender<ScanProgress>>,
) -> Result<()> {
    // Spawn background summary worker — fills in summaries after the fast scan.
    let summary_tx = spawn_summary_worker(
        Arc::clone(&store),
        Arc::clone(&summarizer),
        Arc::clone(&config),
    );
    let root = config.project_root.read().unwrap().clone();
    info!("Starting initial project scan: {}", root);

    {
        let mut p = progress.write().unwrap();
        *p = ProgressState {
            status: "scanning".into(),
            ..Default::default()
        };
    }

    // Spawn a blocking thread that walks the directory and sends paths over a channel.
    // This avoids collecting all paths into a Vec upfront (no memory spike).
    let (path_tx, path_rx) = tokio::sync::mpsc::channel::<String>(256);
    let root_clone = root.clone();
    tokio::task::spawn_blocking(move || {
        WalkBuilder::new(&root_clone)
            .standard_filters(true)
            .add_custom_ignore_filename(".vector-ignore")
            .hidden(true)
            .build()
            .filter_map(|r| r.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
            .filter(is_indexable)
            .for_each(|e| {
                let path = e.path().to_string_lossy().to_string();
                // If the receiver is gone (scan cancelled), stop walking.
                let _ = path_tx.blocking_send(path);
            });
    });

    // Channel for workers to send back their produced records.
    let (record_tx, mut record_rx) =
        tokio::sync::mpsc::channel::<(String, Vec<crate::db::Record>)>(64);

    let token = uuid::Uuid::new_v4().to_string();
    let discovered = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let indexed = Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Convert the mpsc receiver into a stream and process concurrently.
    let path_stream = tokio_stream::wrappers::ReceiverStream::new(path_rx);

    let record_tx_clone = record_tx.clone();
    let progress_clone = Arc::clone(&progress);
    let discovered_clone = Arc::clone(&discovered);

    let worker_task = path_stream.for_each_concurrent(10, |path_str| {
        let config = Arc::clone(&config);
        let store = Arc::clone(&store);
        let embedder = Arc::clone(&embedder);
        let summarizer = Arc::clone(&summarizer);
        let progress = Arc::clone(&progress_clone);
        let record_tx = record_tx_clone.clone();
        let discovered = Arc::clone(&discovered_clone);
        let indexed = Arc::clone(&indexed);
        async move {
            let n = discovered.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            {
                let mut p = progress.write().unwrap();
                p.current_file = path_str.clone();
                p.total_files = n; // dynamically updated as files are discovered
            }

            match indexer::index_file(
                &path_str,
                Arc::clone(&config),
                Arc::clone(&store),
                Arc::clone(&embedder),
                Arc::clone(&summarizer),
            )
            .await
            {
                Ok(records) => {
                    if !records.is_empty() {
                        let _ = record_tx.send((path_str, records)).await;
                    }
                    // Increment indexed count immediately for better UI feedback
                    let c = indexed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let mut p = progress.write().unwrap();
                    p.indexed_files = c;
                }
                Err(e) => {
                    warn!("Failed to index {}: {:?}", path_str, e);
                    let mut p = progress.write().unwrap();
                    p.errors.push(format!("{}: {}", path_str, e));
                }
            }
        }
    });

    // Accumulator task: drain the record channel and flush in batches.
    let store_clone = Arc::clone(&store);

    let accumulator_task = async move {
        let mut batch: Vec<crate::db::Record> = Vec::with_capacity(BATCH_INSERT_THRESHOLD);
        let mut paths_in_batch: Vec<String> = Vec::new();
        let mut deleted_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

        while let Some((path, records)) = record_rx.recv().await {
            // Only delete stale records once per path per scan run.
            if !deleted_paths.contains(&path) {
                if let Err(e) = store_clone.delete_by_path_no_commit(&path).await {
                    warn!("Failed to delete stale records for {}: {:?}", path, e);
                }
                deleted_paths.insert(path.clone());
            }
            batch.extend(records);
            paths_in_batch.push(path);

            if batch.len() >= BATCH_INSERT_THRESHOLD {
                if let Err(e) = store_clone.insert_batch(&batch).await {
                    warn!("Batch insert failed: {:?}", e);
                }
                batch.clear();
                paths_in_batch.clear();
            }
        }

        // Flush remaining records.
        if !batch.is_empty() {
            if let Err(e) = store_clone.insert_batch(&batch).await {
                warn!("Final batch insert failed: {:?}", e);
            }
        }
        // Final Tantivy commit after all bulk operations.
        let _ = store_clone.commit_lexical();
    };

    // Drop the extra sender so the accumulator can detect when all workers are done.
    drop(record_tx);

    // Run workers and accumulator concurrently.
    tokio::join!(worker_task, accumulator_task);

    let final_count = indexed.load(std::sync::atomic::Ordering::Relaxed);
    let total = discovered.load(std::sync::atomic::Ordering::Relaxed);
    info!("Initial scan complete. Indexed {} files.", final_count);

    {
        let mut p = progress.write().unwrap();
        p.status = "Analyzing Relationships".into();
        p.indexed_files = final_count;
        p.total_files = total;
    }

    // Refresh knowledge graph once after all inserts.
    if let Ok(all) = store.get_all_records().await {
        store.graph.populate(&all);
        // Enqueue records that need summarization (function_score > 0.5, no summary yet).
        for r in &all {
            let meta = r.metadata_json();
            let already_summarized = meta["summary"]
                .as_str()
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if already_summarized {
                continue;
            }
            let score = meta["function_score"].as_f64().unwrap_or(0.0) as f32;
            let _ = summary_tx.try_send((r.id.clone(), r.content.clone(), score));
        }
    }

    {
        let mut p = progress.write().unwrap();
        p.status = "Ready".into();
    }

    if let Some(tx) = &progress_tx {
        let _ = tx
            .send(ScanProgress {
                progress_token: token,
                progress: final_count,
                total,
                current_file: String::new(),
                status: "Ready",
            })
            .await;
    }

    Ok(())
}

/// Spawns a low-priority background task that fills in summaries for records
/// that were indexed without one.
///
/// Callers send `(record_id, content, function_score)` tuples. The worker
/// batches up to 10 items or waits 5 seconds, then generates summaries via
/// the local model and patches LanceDB with `update_record_metadata` —
/// no graph rebuild, no BM25 churn.
pub fn spawn_summary_worker(
    store: Arc<Store>,
    summarizer: Arc<crate::llm::summarizer::Summarizer>,
    config: Arc<Config>,
) -> tokio::sync::mpsc::Sender<(String, String, f32)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, String, f32)>(1024);

    tokio::spawn(async move {
        loop {
            // Collect up to 10 items or wait at most 5 seconds.
            let mut batch: Vec<(String, String, f32)> = Vec::with_capacity(10);
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

            loop {
                match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Some(item)) => {
                        batch.push(item);
                        if batch.len() >= 10 {
                            break;
                        }
                    }
                    Ok(None) => return, // channel closed
                    Err(_) => break,    // 5-second timeout
                }
            }

            if batch.is_empty() || !config.feature_toggles.enable_local_llm {
                continue;
            }

            for (record_id, content, score) in batch {
                if score <= 0.5 {
                    continue;
                }
                let summary = match summarizer
                    .summarize_chunk(&content, Arc::clone(&config))
                    .await
                {
                    Ok(s) if !s.is_empty() => s,
                    _ => continue,
                };

                // Fetch current metadata, patch summary field, write back.
                if let Ok(Some(r)) = store.get_record_by_id(&record_id).await {
                    let mut meta = r.metadata_json();
                    meta["summary"] = serde_json::Value::String(summary);
                    let _ = store
                        .update_record_metadata(&record_id, &meta.to_string())
                        .await;
                }
                // Yield to the executor to keep the API responsive
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
    });

    tx
}

/// Determines if a file is suitable for semantic indexing.
/// Uses cached metadata from the `ignore` crate's `DirEntry` to avoid extra syscalls.
fn is_indexable(entry: &DirEntry) -> bool {
    let path = entry.path();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let is_source = matches!(
        ext,
        "go" | "rs" | "js" | "ts" | "tsx" | "php" | "py" | "md" | "pdf"
    );

    if !is_source {
        return false;
    }

    // Use cached metadata from WalkBuilder — avoids a redundant stat() syscall.
    if let Ok(metadata) = entry.metadata() {
        let size = metadata.len();
        let limit = if ext == "pdf" { 10_000_000 } else { 1_000_000 };
        if size > limit {
            warn!(
                "Skipping oversized file: {} ({} bytes)",
                path.display(),
                size
            );
            return false;
        }
    }

    true
}
