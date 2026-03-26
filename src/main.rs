mod config;
mod db;
mod indexer;
mod llm;
mod mcp;

use std::sync::Arc;
use anyhow::{Result, Context};
use tracing::info;

use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Structured logging (JSON to stderr, respects RUST_LOG env filter).
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // 2. Load configuration from env / .env file.
    let cfg = config::Config::load()?;
    info!(
        project_root = %cfg.project_root,
        db_path = %cfg.db_path.display(),
        model = %cfg.model_name,
        local_llm = cfg.feature_toggles.enable_local_llm,
        live_indexing = cfg.feature_toggles.enable_live_indexing,
        "configuration loaded"
    );

    // 3. Connect to LanceDB and open/create tables.
    let db_uri = cfg.db_path.display().to_string();
    let store = db::connect_store(&db_uri, cfg.dimension).await?;
    info!("LanceDB connected — tables initialised");

    // 4. Initialise AI components.
    // Use the model path from config.
    let embedder = Arc::new(Embedder::new(&cfg)?);
    let summarizer = Arc::new(Summarizer::new(&cfg)?);

    let store = Arc::new(store);
    let config = Arc::new(cfg);

    // 5. Initial Scan & Background Watcher (if enabled).
    // Perform an initial project-wide scan to ensure we have a baseline index.
    let _ = crate::indexer::scanner::scan_project(
        Arc::clone(&config),
        Arc::clone(&store),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
    ).await;

    indexer::watcher::start_watcher(
        Arc::clone(&config),
        Arc::clone(&store),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
    ).await.context("Starting background watcher")?;

    // 6. Start MCP server.
    let server = Arc::new(mcp::server::Server::new(
        Arc::clone(&store),
        Arc::clone(&config),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
    ));

    let port_str = config.api_port.clone();
    let sse_server = if let Ok(port) = port_str.parse::<u16>() {
        let server_for_sse = Arc::clone(&server);
        Some(tokio::spawn(async move {
            if let Err(e) = mcp::sse::start_sse_server(server_for_sse, port).await {
                tracing::error!(err = %e, "SSE server failed");
            }
        }))
    } else {
        None
    };

    info!("vector-mcp-rust ready ✓");

    // Standard MCP transport via stdio.
    server.run().await?;

    // Graceful shutdown of SSE server if it was running.
    if let Some(task) = sse_server {
        task.abort();
    }

    Ok(())
}
