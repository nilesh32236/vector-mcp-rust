mod config;
mod db;
mod indexer;
mod llm;
mod mcp;

use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::info;

use crate::config::Config;
use crate::db::Store;
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;

fn setup_logging() {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

async fn init_components(cfg: &Config) -> Result<(Arc<Store>, Arc<Embedder>, Arc<Summarizer>)> {
    let db_uri = cfg.db_path.display().to_string();
    let store = db::connect_store(&db_uri, cfg.dimension).await?;
    info!("LanceDB connected — tables initialised");

    let embedder = Arc::new(Embedder::new(cfg)?);
    let summarizer = Arc::new(Summarizer::new(cfg)?);

    Ok((Arc::new(store), embedder, summarizer))
}

async fn start_background_tasks(
    config: Arc<Config>,
    store: Arc<Store>,
    embedder: Arc<Embedder>,
    summarizer: Arc<Summarizer>,
) -> Result<()> {
    let config_clone = Arc::clone(&config);
    let store_clone = Arc::clone(&store);
    let embedder_clone = Arc::clone(&embedder);
    let summarizer_clone = Arc::clone(&summarizer);

    tokio::spawn(async move {
        let _ = crate::indexer::scanner::scan_project(
            config_clone,
            store_clone,
            embedder_clone,
            summarizer_clone,
        )
        .await;
    });

    indexer::watcher::start_watcher(
        Arc::clone(&config),
        Arc::clone(&store),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
    )
    .await
    .context("Starting background watcher")?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Structured logging (JSON to stderr, respects RUST_LOG env filter).
    setup_logging();

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
    // 4. Initialise AI components.
    let (store, embedder, summarizer) = init_components(&cfg).await?;
    let config = Arc::new(cfg);

    // 5. Initial Scan & Background Watcher (if enabled).
    start_background_tasks(
        Arc::clone(&config),
        Arc::clone(&store),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
    )
    .await?;

    // 6. Start MCP server.
    let server = Arc::new(mcp::server::Server::new(
        Arc::clone(&store),
        Arc::clone(&config),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
    ));

    let port_str = config.api_port.clone();
    if let Ok(port) = port_str.parse::<u16>() {
        let server_for_sse = Arc::clone(&server);
        info!("vector-mcp-rust ready ✓ (SSE transport on port {})", port);

        // Block on the SSE server, allowing Ctrl-C to terminate it cleanly.
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl-C, shutting down gracefully...");
            }
            res = mcp::sse::start_sse_server(server_for_sse, port) => {
                if let Err(e) = res {
                    tracing::error!(err = %e, "SSE server failed");
                }
            }
        }
    } else {
        info!("vector-mcp-rust ready ✓ (stdio transport)");
        // Standard MCP transport via stdio.
        server.run().await?;
    }

    Ok(())
}
