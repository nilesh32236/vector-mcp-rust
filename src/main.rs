//! vector-mcp-rust — entry point.
//!
//! ## Master / Slave mode
//!
//! On startup the process attempts to bind the Unix socket at `DAEMON_SOCKET`
//! (default `~/.local/share/vector-mcp-rust/daemon.sock`).
//!
//! - **Master** (socket free): loads the ONNX model, opens LanceDB, starts the
//!   daemon RPC server, the file watcher, the MCP SSE server, and the HTTP API.
//! - **Slave** (socket taken): skips heavy initialisation, connects to the master
//!   via [`daemon::slave::RemoteEmbedder`] / [`daemon::slave::RemoteStore`], and
//!   starts a lightweight MCP server that delegates all AI/DB work to the master.

mod api;
mod config;
mod daemon;
mod db;
mod indexer;
mod llm;
mod lsp;
mod mcp;
mod mutation;
mod security;

use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::info;

use crate::config::Config;
use crate::db::Store;
use crate::llm::embedding::Embedder;
use crate::llm::summarizer::Summarizer;

// ---------------------------------------------------------------------------
// Logging & Telemetry
// ---------------------------------------------------------------------------

/// Initialise structured logging and OpenTelemetry.
fn setup_logging(
    log_path: &std::path::Path,
) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_subscriber::prelude::*;

    // Ensure the log directory exists.
    if let Some(dir) = log_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }

    let file_appender = tracing_appender::rolling::never(
        log_path.parent().unwrap_or(std::path::Path::new(".")),
        log_path
            .file_name()
            .unwrap_or(std::ffi::OsStr::new("mcp.log")),
    );
    let (non_blocking_file, guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    let file_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(non_blocking_file);

    let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(file_layer)
        .with(stderr_layer);

    if std::env::var("ENABLE_OTEL").is_ok() {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint("http://localhost:4317")
            .build()?;

        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(
                Resource::builder()
                    .with_service_name("vector-mcp-rust")
                    .build(),
            )
            .build();

        let tracer = provider.tracer("vector-mcp-rust");
        let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
        registry.with(telemetry).init();
    } else {
        registry.init();
    }

    Ok(guard)
}

// ---------------------------------------------------------------------------
// Component initialisation
// ---------------------------------------------------------------------------

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
    progress: Arc<std::sync::RwLock<crate::indexer::scanner::ProgressState>>,
) -> Result<Option<notify::RecommendedWatcher>> {
    // Initial scan.
    let (c, s, e, su, p) = (
        Arc::clone(&config),
        Arc::clone(&store),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
        Arc::clone(&progress),
    );
    tokio::spawn(async move {
        let _ = crate::indexer::scanner::scan_project(c, s, e, su, p, None).await;
    });

    // File watcher — caller must hold the returned watcher alive.
    let watcher = indexer::watcher::start_watcher(
        Arc::clone(&config),
        Arc::clone(&store),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
    )
    .await
    .context("Starting background watcher")?;

    Ok(watcher)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load configuration first so we know the log path.
    let cfg = config::Config::load()?;

    // 2. Structured logging → file (stderr stays clean for MCP JSON-RPC).
    let _log_guard = setup_logging(&cfg.log_path)?;

    info!(
        project_root = %*cfg.project_root.read().unwrap(),
        db_path      = %cfg.db_path.display(),
        model        = %cfg.model_name,
        log          = %cfg.log_path.display(),
        "configuration loaded"
    );

    // 2. Resolve daemon socket path.
    let socket_path = std::env::var("DAEMON_SOCKET")
        .unwrap_or_else(|_| cfg.data_dir.join("daemon.sock").display().to_string());

    // 3. Detect master / slave mode.
    let is_slave = daemon::slave::master_is_running(&socket_path).await;

    if is_slave {
        run_slave(cfg, socket_path).await
    } else {
        run_master(cfg, socket_path).await
    }
}

// ---------------------------------------------------------------------------
// Master mode
// ---------------------------------------------------------------------------

async fn run_master(cfg: Config, socket_path: String) -> Result<()> {
    info!("Starting in MASTER mode");

    // Initialise heavy components.
    let (store, embedder, summarizer) = init_components(&cfg).await?;
    let config = Arc::new(cfg);

    // Shared indexing progress map.
    let progress = Arc::new(std::sync::RwLock::new(
        crate::indexer::scanner::ProgressState::default(),
    ));

    // Background indexing channel (wired to daemon + watcher).
    let (index_tx, mut index_rx) = tokio::sync::mpsc::channel::<String>(256);

    // Start daemon RPC server.
    let daemon_server = Arc::new(daemon::master::MasterServer::new(
        &socket_path,
        Arc::clone(&store),
        Arc::clone(&embedder),
        index_tx.clone(),
        Arc::clone(&progress),
    ));
    daemon_server.start().await?;
    info!(socket = %socket_path, "Daemon RPC server started");

    // Background worker: drain the index queue.
    {
        let cfg2 = Arc::clone(&config);
        let store2 = Arc::clone(&store);
        let emb2 = Arc::clone(&embedder);
        let sum2 = Arc::clone(&summarizer);
        let prog2 = Arc::clone(&progress);
        tokio::spawn(async move {
            while let Some(path) = index_rx.recv().await {
                let _ = crate::indexer::scanner::scan_project(
                    Arc::clone(&cfg2),
                    Arc::clone(&store2),
                    Arc::clone(&emb2),
                    Arc::clone(&sum2),
                    Arc::clone(&prog2),
                    None,
                )
                .await;
                let _ = path; // path is the trigger; scan_project uses project_root
            }
        });
    }

    // Initial scan + file watcher — hold the watcher alive for the process lifetime.
    let _watcher = start_background_tasks(
        Arc::clone(&config),
        Arc::clone(&store),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
        Arc::clone(&progress),
    )
    .await?;

    // Watcher reload channel — hold each new watcher alive by storing it.
    let (reload_tx, mut reload_rx) = tokio::sync::mpsc::channel::<String>(10);
    {
        let cfg3 = Arc::clone(&config);
        let store3 = Arc::clone(&store);
        let emb3 = Arc::clone(&embedder);
        let sum3 = Arc::clone(&summarizer);
        tokio::spawn(async move {
            let mut _active_watcher: Option<notify::RecommendedWatcher> = None;
            while reload_rx.recv().await.is_some() {
                _active_watcher = indexer::watcher::start_watcher(
                    Arc::clone(&cfg3),
                    Arc::clone(&store3),
                    Arc::clone(&emb3),
                    Arc::clone(&sum3),
                )
                .await
                .ok()
                .flatten();
            }
        });
    }

    // MCP server.
    let server = Arc::new(mcp::server::Server::new(
        Arc::clone(&store),
        Arc::clone(&config),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
        reload_tx,
    ));

    start_servers(
        server, store, embedder, summarizer, config, progress, index_tx,
    )
    .await
}

// ---------------------------------------------------------------------------
// Slave mode
// ---------------------------------------------------------------------------

async fn run_slave(cfg: Config, socket_path: String) -> Result<()> {
    info!(socket = %socket_path, "Starting in SLAVE mode — delegating to master");

    let config = Arc::new(cfg);
    let progress = Arc::new(std::sync::RwLock::new(
        crate::indexer::scanner::ProgressState::default(),
    ));
    let (index_tx, _index_rx) = tokio::sync::mpsc::channel::<String>(1);

    // Slaves do not open LanceDB or load ONNX — they delegate everything.
    // We still need a local Store for the MCP server struct; use a lightweight
    // in-process store connected to the same LanceDB path (read-only queries
    // that aren't performance-critical). Heavy ops go via the daemon.
    let db_uri = config.db_path.display().to_string();
    let store = Arc::new(db::connect_store(&db_uri, config.dimension).await?);

    // NOTE: Embedder is loaded locally until the Server struct is refactored to
    // accept a trait object. RemoteEmbedder exists in daemon::slave for that future work.
    let embedder = Arc::new(Embedder::new(&config)?);
    let summarizer = Arc::new(Summarizer::new(&config)?);

    let (reload_tx, _reload_rx) = tokio::sync::mpsc::channel::<String>(10);

    let server = Arc::new(mcp::server::Server::new(
        Arc::clone(&store),
        Arc::clone(&config),
        Arc::clone(&embedder),
        Arc::clone(&summarizer),
        reload_tx,
    ));

    // Slaves do NOT run the file watcher (master owns indexing).
    start_servers(
        server, store, embedder, summarizer, config, progress, index_tx,
    )
    .await
}

// ---------------------------------------------------------------------------
// Shared server startup
// ---------------------------------------------------------------------------

async fn start_servers(
    server: Arc<mcp::server::Server>,
    store: Arc<Store>,
    embedder: Arc<Embedder>,
    summarizer: Arc<Summarizer>,
    config: Arc<Config>,
    progress: Arc<std::sync::RwLock<crate::indexer::scanner::ProgressState>>,
    index_tx: tokio::sync::mpsc::Sender<String>,
) -> Result<()> {
    let port_str = config.api_port.clone();

    if let Ok(port) = port_str.parse::<u16>() {
        let api_port = port + 1;
        let api_state = Arc::new(api::ApiState {
            store: Arc::clone(&store),
            embedder: Arc::clone(&embedder),
            summarizer: Arc::clone(&summarizer),
            config: Arc::clone(&config),
            index_tx,
            rate_limiter: Arc::clone(&server.rate_limiter),
            progress,
            version: env!("CARGO_PKG_VERSION"),
        });

        info!(mcp_port = port, api_port, "vector-mcp-rust ready ✓");

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl-C received — shutting down");
            }
            res = mcp::sse::start_sse_server(Arc::clone(&server), port) => {
                if let Err(e) = res { tracing::error!(err = %e, "SSE server failed"); }
            }
            res = api::start_api_server(api_state, api_port) => {
                if let Err(e) = res { tracing::error!(err = %e, "API server failed"); }
            }
        }
    } else {
        info!("vector-mcp-rust ready ✓ (stdio transport)");
        server.run().await?;
    }

    Ok(())
}
