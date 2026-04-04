use anyhow::Result;
use futures::{StreamExt, stream};
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

/// Scans the entire project root and indexes all relevant source files.
/// Respects `.gitignore` and skip patterns via the `ignore` crate.
///
/// `progress_tx` — optional channel for `$/progress` notifications.
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

    // Collect all indexable paths first so we know the total.
    let all_paths: Vec<_> = WalkBuilder::new(&root)
        .standard_filters(true)
        .add_custom_ignore_filename(".vector-ignore")
        .hidden(true)
        .build()
        .filter_map(|r| r.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .filter(is_indexable)
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();

    let total = all_paths.len() as u64;
    {
        progress.write().unwrap().total_files = total;
    }

    let token = uuid::Uuid::new_v4().to_string();
    let count = Arc::new(std::sync::atomic::AtomicU64::new(0));

    stream::iter(all_paths)
        .for_each_concurrent(10, |path_str| {
            let config = Arc::clone(&config);
            let store = Arc::clone(&store);
            let embedder = Arc::clone(&embedder);
            let summarizer = Arc::clone(&summarizer);
            let progress = Arc::clone(&progress);
            let progress_tx = progress_tx.clone();
            let token = token.clone();
            let count = Arc::clone(&count);
            async move {
                info!("Scanning: {}", path_str);
                {
                    progress.write().unwrap().current_file = path_str.clone();
                }

                if let Err(e) = indexer::index_file(
                    &path_str,
                    Arc::clone(&config),
                    Arc::clone(&store),
                    Arc::clone(&embedder),
                    Arc::clone(&summarizer),
                )
                .await
                {
                    warn!("Failed to index {}: {:?}", path_str, e);
                    progress
                        .write()
                        .unwrap()
                        .errors
                        .push(format!("{}: {}", path_str, e));
                } else {
                    let c = count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    progress.write().unwrap().indexed_files = c;

                    if let Some(tx) = &progress_tx
                        && (c.is_multiple_of(10) || c == total)
                    {
                        let _ = tx
                            .send(ScanProgress {
                                progress_token: token.clone(),
                                progress: c,
                                total,
                                current_file: path_str.clone(),
                                status: "indexing",
                            })
                            .await;
                    }
                }
            }
        })
        .await;

    let final_count = count.load(std::sync::atomic::Ordering::Relaxed);
    info!("Initial scan complete. Indexed {} files.", final_count);
    progress.write().unwrap().status = "complete".into();

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
