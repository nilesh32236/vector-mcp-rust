use anyhow::Result;
use ignore::WalkBuilder;
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

use crate::config::Config;
use crate::db::Store;
use crate::indexer;
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;

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
    progress: Arc<dashmap::DashMap<String, serde_json::Value>>,
    progress_tx: Option<tokio::sync::mpsc::Sender<ScanProgress>>,
) -> Result<()> {
    let root = config.project_root.read().unwrap().clone();
    info!("Starting initial project scan: {}", root);

    progress.clear();
    progress.insert("status".to_string(), serde_json::json!("scanning"));
    progress.insert("current_file".to_string(), serde_json::json!(""));
    progress.insert("indexed_files".to_string(), serde_json::json!(0));
    progress.insert("errors".to_string(), serde_json::json!(Vec::<String>::new()));

    // Collect all indexable paths first so we know the total.
    let all_paths: Vec<_> = WalkBuilder::new(&root)
        .standard_filters(true)
        .add_custom_ignore_filename(".vector-ignore")
        .hidden(true)
        .build()
        .filter_map(|r| r.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .filter(|e| is_indexable(e.path()))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();

    let total = all_paths.len() as u64;
    progress.insert("total_files".to_string(), serde_json::json!(total));
    
    let token = uuid::Uuid::new_v4().to_string();
    let mut count: u64 = 0;

    for path_str in all_paths {
        info!("Scanning: {}", path_str);
        progress.insert("current_file".to_string(), serde_json::json!(&path_str));

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
            if let Some(mut errs) = progress.get_mut("errors")
                && let Some(arr) = errs.value_mut().as_array_mut() {
                    arr.push(serde_json::json!(format!("{}: {}", path_str, e)));
                }
        } else {
            count += 1;
            progress.insert("indexed_files".to_string(), serde_json::json!(count));
        }

        // Emit progress notification every 10 files or on the last file.
        if let Some(tx) = &progress_tx
            && (count.is_multiple_of(10) || count == total) {
                let _ = tx
                    .send(ScanProgress {
                        progress_token: token.clone(),
                        progress: count,
                        total,
                        current_file: path_str.clone(),
                        status: "indexing",
                    })
                    .await;
            }
    }

    info!("Initial scan complete. Indexed {} files.", count);
    progress.insert("status".to_string(), serde_json::json!("complete"));

    // Final notification.
    if let Some(tx) = &progress_tx {
        let _ = tx
            .send(ScanProgress {
                progress_token: token,
                progress: count,
                total,
                current_file: String::new(),
                status: "complete",
            })
            .await;
    }

    Ok(())
}

/// Determines if a file is suitable for semantic indexing.
fn is_indexable(path: &Path) -> bool {
    // 1. Check extension
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let is_source = matches!(
        ext,
        "go" | "rs" | "js" | "ts" | "tsx" | "php" | "py" | "md" | "pdf"
    );

    if !is_source {
        return false;
    }

    // 2. Avoid oversized files (e.g. 1MB limit for safety)
    if let Ok(metadata) = std::fs::metadata(path)
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
