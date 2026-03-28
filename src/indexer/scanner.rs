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

/// Scans the entire project root and indexes all relevant source files.
/// Respects `.gitignore` and skip patterns via the `ignore` crate.
pub async fn scan_project(
    config: Arc<Config>,
    store: Arc<Store>,
    embedder: Arc<Embedder>,
    summarizer: Arc<Summarizer>,
    progress: Arc<dashmap::DashMap<String, serde_json::Value>>,
) -> Result<()> {
    info!(
        "Starting initial project scan: {}",
        config.project_root.read().unwrap().clone()
    );
    progress.clear();
    progress.insert("status".to_string(), serde_json::json!("scanning"));
    progress.insert("current_file".to_string(), serde_json::json!(""));
    progress.insert("indexed_files".to_string(), serde_json::json!(0));
    progress.insert(
        "errors".to_string(),
        serde_json::json!(Vec::<String>::new()),
    );

    let walker = WalkBuilder::new(&*config.project_root.read().unwrap())
        .standard_filters(true) // respects .gitignore, etc.
        .hidden(true)
        .build();

    let mut count = 0;
    for result in walker {
        let entry = match result {
            Ok(entry) => entry,
            Err(e) => {
                warn!("Scanner error: {}", e);
                continue;
            }
        };

        if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            let path = entry.path();
            if is_indexable(path) {
                let path_str = path.to_string_lossy().to_string();
                info!("Scanning: {}", path_str);
                progress.insert(
                    "current_file".to_string(),
                    serde_json::json!(path_str.clone()),
                );

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
                    if let Some(mut errs) = progress.get_mut("errors") {
                        if let Some(arr) = errs.value_mut().as_array_mut() {
                            arr.push(serde_json::json!(format!("{}: {}", path_str, e)));
                        }
                    }
                } else {
                    count += 1;
                    progress.insert("indexed_files".to_string(), serde_json::json!(count));
                }
            }
        }
    }

    info!("Initial scan complete. Indexed {} files.", count);
    progress.insert("status".to_string(), serde_json::json!("complete"));
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
