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
                }
                Err(e) => {
                    warn!("Failed to index {}: {:?}", path_str, e);
                    progress
                        .write()
                        .unwrap()
                        .errors
                        .push(format!("{}: {}", path_str, e));
                }
            }
        }
    });

    // Accumulator task: drain the record channel and flush in batches.
    let store_clone = Arc::clone(&store);
    let progress_acc = Arc::clone(&progress);
    let indexed_acc = Arc::clone(&indexed);
    let token_clone = token.clone();
    let total_ref = Arc::clone(&discovered);
    let progress_tx_clone = progress_tx.clone();

    let accumulator_task = async move {
        let mut batch: Vec<crate::db::Record> = Vec::with_capacity(BATCH_INSERT_THRESHOLD);
        let mut paths_in_batch: Vec<String> = Vec::new();

        while let Some((path, records)) = record_rx.recv().await {
            // Delete stale records for this path before accumulating new ones.
            if let Err(e) = store_clone.delete_by_path(&path).await {
                warn!("Failed to delete stale records for {}: {:?}", path, e);
            }
            batch.extend(records);
            paths_in_batch.push(path);

            if batch.len() >= BATCH_INSERT_THRESHOLD {
                if let Err(e) = store_clone.insert_batch(&batch).await {
                    warn!("Batch insert failed: {:?}", e);
                }
                let c = indexed_acc.fetch_add(
                    paths_in_batch.len() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                ) + paths_in_batch.len() as u64;
                let total = total_ref.load(std::sync::atomic::Ordering::Relaxed);
                progress_acc.write().unwrap().indexed_files = c;

                if let Some(tx) = &progress_tx_clone
                    && c.is_multiple_of(10)
                {
                    let _ = tx
                        .send(ScanProgress {
                            progress_token: token_clone.clone(),
                            progress: c,
                            total,
                            current_file: paths_in_batch.last().cloned().unwrap_or_default(),
                            status: "indexing",
                        })
                        .await;
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
            let c = indexed_acc.fetch_add(
                paths_in_batch.len() as u64,
                std::sync::atomic::Ordering::Relaxed,
            ) + paths_in_batch.len() as u64;
            progress_acc.write().unwrap().indexed_files = c;
        }
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
        p.status = "complete".into();
        p.indexed_files = final_count;
        p.total_files = total;
    }

    // Refresh knowledge graph once after all inserts.
    if let Ok(all) = store.get_all_records().await {
        store.graph.populate(&all);
    }

    if let Some(tx) = &progress_tx {
        let _ = tx
            .send(ScanProgress {
                progress_token: token,
                progress: final_count,
                total,
                current_file: String::new(),
                status: "complete",
            })
            .await;
    }

    Ok(())
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
    if let Ok(metadata) = entry.metadata()
        && metadata.len() > 1_000_000
        && ext != "pdf"
    {
        warn!(
            "Skipping oversized file: {} ({} bytes)",
            path.display(),
            metadata.len()
        );
        return false;
    }

    true
}
