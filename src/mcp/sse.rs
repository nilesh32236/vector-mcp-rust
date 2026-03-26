//! SSE transport for MCP.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use axum::response::sse;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{sse::Event, Sse},
    routing::{get, post},
    Json, Router,
};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use uuid::Uuid;
use futures::stream::Stream;
use tokio_stream::StreamExt;

use super::server::Server;

// ---------------------------------------------------------------------------
// SSE Session Manager
// ---------------------------------------------------------------------------

type SessionSender = mpsc::UnboundedSender<Event>;

pub struct SseManager {
    sessions: Mutex<HashMap<String, SessionSender>>,
    server: Arc<Server>,
}

impl SseManager {
    pub fn new(server: Arc<Server>) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            server,
        }
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct MessageQuery {
    session_id: String,
}

/// GET /sse — Start the SSE connection.
async fn sse_handler(
    State(manager): State<Arc<SseManager>>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = mpsc::unbounded_channel();

    {
        let mut sessions = manager.sessions.lock().await;
        sessions.insert(session_id.clone(), tx.clone());
    }

    info!(session = %session_id, "new SSE session");

    // Send the endpoint event as per MCP spec.
    // The client will use this URL to POST messages.
    let endpoint_url = format!("/message?session_id={}", session_id);
    let _ = tx.send(Event::default().event("endpoint").data(endpoint_url));

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(Ok);

    Sse::new(stream)
}

/// POST /message — Receive JSON-RPC messages.
async fn message_handler(
    State(manager): State<Arc<SseManager>>,
    Query(query): Query<MessageQuery>,
    body: String,
) -> StatusCode {
    let sender = {
        let sessions = manager.sessions.lock().await;
        sessions.get(&query.session_id).cloned()
    };

    let tx = match sender {
        Some(t) => t,
        None => {
            warn!(session = %query.session_id, "message posted to invalid session");
            return StatusCode::NOT_FOUND;
        }
    };

    let server = Arc::clone(&manager.server);
    let session_id = query.session_id.clone();

    // Process the message asynchronously and send the response back via SSE.
    tokio::spawn(async move {
        let response = server.process_message(&body).await;
        let response_json = serde_json::to_string(&response).unwrap_or_default();
        
        let event = Event::default().event("message").data(response_json);
        if let Err(_) = tx.send(event) {
            warn!(session = %session_id, "failed to send response to SSE stream (client disconnected)");
        }
    });

    StatusCode::ACCEPTED
}

// ---------------------------------------------------------------------------
// Server Bootstrap
// ---------------------------------------------------------------------------

pub async fn start_sse_server(server: Arc<Server>, port: u16) -> anyhow::Result<()> {
    let manager = Arc::new(SseManager::new(server));

    let app = Router::new()
        .route("/sse", get(sse_handler))
        .route("/message", post(message_handler))
        .layer(CorsLayer::permissive())
        .with_state(manager);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    
    info!("SSE server listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}
