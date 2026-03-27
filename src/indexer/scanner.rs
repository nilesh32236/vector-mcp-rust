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
) -> Result<()> {
    info!("Starting initial project scan: {}", config.project_root);

    let walker = WalkBuilder::new(&config.project_root)
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
                } else {
                    count += 1;
                }
            }
        }
    }

    info!("Initial scan complete. Indexed {} files.", count);
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
    if let Ok(metadata) = std::fs::metadata(path) {
        if metadata.len() > 1_000_000 && ext != "pdf" {
            warn!(
                "Skipping oversized file: {} ({} bytes)",
                path.display(),
                metadata.len()
            );
            return false;
        }
    }

    true
}
