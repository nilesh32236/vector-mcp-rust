use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use notify::{Watcher, RecursiveMode, Event, EventKind};
use tokio::sync::mpsc;
use tracing::{info, error};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::time::Instant;

use crate::config::Config;
use crate::db::Store;
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;
use crate::indexer;

/// Starts the background CDC file watcher.
pub async fn start_watcher(
    config: Arc<Config>,
    db: Arc<Store>,
    embedder: Arc<Embedder>,
    summarizer: Arc<Summarizer>,
) -> Result<()> {
    if !config.feature_toggles.enable_live_indexing {
        info!("Live indexing disabled via config");
        return Ok(());
    }

    info!("Initializing background CDC watcher on: {}", config.project_root);

    let (tx, mut rx) = mpsc::channel::<PathBuf>(100);

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        match res {
            Ok(event) => {
                if is_interesting_event(&event) {
                    for path in event.paths {
                        let _ = tx.blocking_send(path);
                    }
                }
            }
            Err(e) => error!("watch error: {:?}", e),
        }
    })?;

    watcher.watch(std::path::Path::new(&config.project_root), RecursiveMode::Recursive)?;

    tokio::spawn(async move {
        let mut pending: HashMap<PathBuf, Instant> = HashMap::new();
        let debounce_duration = Duration::from_millis(500);

        loop {
            tokio::select! {
                Some(path) = rx.recv() => {
                    pending.insert(path, Instant::now() + debounce_duration);
                }
                
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    let now = Instant::now();
                    let ready_paths: Vec<PathBuf> = pending
                        .iter()
                        .filter(|(_, deadline)| now >= **deadline)
                        .map(|(path, _)| path.clone())
                        .collect();

                    for path in ready_paths {
                        pending.remove(&path);
                        
                        let path_str = path.to_string_lossy().to_string();
                        info!("CDC: Re-indexing changed file: {}", path_str);
                        
                        let config = Arc::clone(&config);
                        let db = Arc::clone(&db);
                        let embedder = Arc::clone(&embedder);
                        let summarizer = Arc::clone(&summarizer);
                        
                        tokio::spawn(async move {
                            if let Err(e) = indexer::index_file(&path_str, config, db, embedder, summarizer).await {
                                error!("Failed to re-index {}: {:?}", path_str, e);
                            }
                        });
                    }
                }
            }
        }
    });

    Box::leak(Box::new(watcher)); 

    Ok(())
}

fn is_interesting_event(event: &Event) -> bool {
    match event.kind {
        EventKind::Modify(_) | EventKind::Create(_) => {
            event.paths.iter().any(|p| {
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                matches!(ext, "go" | "rs" | "js" | "ts" | "tsx" | "php" | "py" | "md")
            })
        }
        _ => false,
    }
}
