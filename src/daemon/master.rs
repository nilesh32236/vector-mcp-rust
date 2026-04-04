//! Master daemon — Unix socket RPC server.
//!
//! Owns the [`Embedder`] and [`Store`] and serves requests from slave instances.
//! Started once at system boot (e.g. via a systemd unit) and kept alive.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{error, info, warn};

use crate::db::{Record, Store};
use crate::llm::embedding::Embedder;

// ---------------------------------------------------------------------------
// Wire types — request / response envelope
// ---------------------------------------------------------------------------

/// Every request sent by a slave over the Unix socket.
#[derive(Debug, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum Request {
    /// Embed a single text string.
    Embed { text: String },
    /// Embed a query string (applies Nomic prefix when relevant).
    EmbedQuery { text: String },
    /// Hybrid search: vector + BM25 RRF fusion.
    HybridSearch {
        query: String,
        vector: Vec<f32>,
        limit: usize,
    },
    /// Enqueue a path for background indexing.
    IndexProject { path: String },
    /// Retrieve live indexing progress.
    GetProgress,
    /// Insert records into the vector store.
    Insert { records: Vec<Record> },
    /// Delete all records whose path matches the prefix.
    DeleteByPath { path: String },
    /// Retrieve all records (used by slaves for full-codebase analysis).
    GetAllRecords,
}

/// Every response sent by the master.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Response {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Response {
    fn ok(result: impl Serialize) -> Self {
        Self {
            ok: true,
            result: Some(serde_json::to_value(result).unwrap_or(serde_json::Value::Null)),
            error: None,
        }
    }

    fn err(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            result: None,
            error: Some(msg.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// MasterServer
// ---------------------------------------------------------------------------

/// The master daemon — listens on a Unix socket and dispatches RPC requests.
pub struct MasterServer {
    socket_path: String,
    store: Arc<Store>,
    embedder: Arc<Embedder>,
    /// Channel for enqueuing paths to the background indexer.
    index_tx: tokio::sync::mpsc::Sender<String>,
    /// Live indexing progress map (shared with the scanner).
    progress: Arc<dashmap::DashMap<String, serde_json::Value>>,
}

impl MasterServer {
    /// Create a new master server.
    ///
    /// `index_tx` should be connected to the background indexing worker.
    pub fn new(
        socket_path: impl Into<String>,
        store: Arc<Store>,
        embedder: Arc<Embedder>,
        index_tx: tokio::sync::mpsc::Sender<String>,
        progress: Arc<dashmap::DashMap<String, serde_json::Value>>,
    ) -> Self {
        Self {
            socket_path: socket_path.into(),
            store,
            embedder,
            index_tx,
            progress,
        }
    }

    /// Start listening. Returns immediately; spawns an accept loop in the background.
    ///
    /// If a stale socket file exists it is removed first. If another master is
    /// already listening the function returns an error (slave mode should be used).
    pub async fn start(self: Arc<Self>) -> Result<()> {
        let path = &self.socket_path;

        // Detect a live master — if we can connect, we are a slave.
        if tokio::net::UnixStream::connect(path).await.is_ok() {
            anyhow::bail!("master already running on {path}");
        }

        // Remove stale socket file.
        let _ = tokio::fs::remove_file(path).await;

        let listener =
            UnixListener::bind(path).with_context(|| format!("binding Unix socket at {path}"))?;

        info!(socket = %path, "Master daemon listening");

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let svc = Arc::clone(&self);
                        tokio::spawn(async move {
                            if let Err(e) = svc.handle_connection(stream).await {
                                warn!("daemon connection error: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        error!("daemon accept error: {e}");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Connection handler
    // -----------------------------------------------------------------------

    async fn handle_connection(&self, stream: UnixStream) -> Result<()> {
        let (reader, mut writer) = stream.into_split();
        let mut lines = BufReader::new(reader).lines();

        while let Some(line) = lines.next_line().await? {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            let response = match serde_json::from_str::<Request>(&line) {
                Ok(req) => self.dispatch(req).await,
                Err(e) => Response::err(format!("invalid request: {e}")),
            };

            let mut json = serde_json::to_string(&response)?;
            json.push('\n');
            writer.write_all(json.as_bytes()).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Request dispatcher
    // -----------------------------------------------------------------------

    async fn dispatch(&self, req: Request) -> Response {
        match req {
            Request::Embed { text } => match self.embedder.embed_text(&text) {
                Ok(v) => Response::ok(serde_json::json!({ "embedding": v })),
                Err(e) => Response::err(e.to_string()),
            },

            Request::EmbedQuery { text } => match self.embedder.embed_query(&text) {
                Ok(v) => Response::ok(serde_json::json!({ "embedding": v })),
                Err(e) => Response::err(e.to_string()),
            },

            Request::HybridSearch {
                query,
                vector,
                limit,
            } => match self.store.hybrid_search(vector, &query, limit, None).await {
                Ok(records) => Response::ok(serde_json::json!({ "records": records })),
                Err(e) => Response::err(e.to_string()),
            },

            Request::IndexProject { path } => match self.index_tx.try_send(path) {
                Ok(_) => Response::ok(serde_json::json!({ "queued": true })),
                Err(_) => Response::err("index queue full"),
            },

            Request::GetProgress => {
                let map: std::collections::HashMap<String, serde_json::Value> = self
                    .progress
                    .iter()
                    .map(|e| (e.key().clone(), e.value().clone()))
                    .collect();
                Response::ok(map)
            }

            Request::Insert { records } => match self.store.upsert_records(records).await {
                Ok(_) => Response::ok(serde_json::json!({ "inserted": true })),
                Err(e) => Response::err(e.to_string()),
            },

            Request::DeleteByPath { path } => match self.store.delete_by_path(&path).await {
                Ok(_) => Response::ok(serde_json::json!({ "deleted": true })),
                Err(e) => Response::err(e.to_string()),
            },

            Request::GetAllRecords => match self.store.get_all_records().await {
                Ok(records) => Response::ok(serde_json::json!({ "records": records })),
                Err(e) => Response::err(e.to_string()),
            },
        }
    }
}
