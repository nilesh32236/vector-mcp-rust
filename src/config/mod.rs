use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Feature Toggles — Low-Resource Mode support
// ---------------------------------------------------------------------------

/// Controls optional subsystems that can be disabled on constrained hardware.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureToggles {
    /// When `true`, the server loads and uses a local LLM for embeddings and
    /// summarisation via `LlamaEngine`.
    pub enable_local_llm: bool,
    /// When `true`, the file-watcher triggers live re-indexing on save.
    pub enable_live_indexing: bool,
}

// ---------------------------------------------------------------------------
// Application Configuration
// ---------------------------------------------------------------------------

/// Central configuration resolved once at startup and shared immutably via
/// `Arc<Config>` with every subsystem.
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub project_root: std::sync::RwLock<String>,
    pub data_dir: PathBuf,
    pub db_path: PathBuf,
    pub models_dir: PathBuf,
    pub log_path: PathBuf,
    /// Retained for future extensibility; not used to drive model loading.
    pub model_name: String,
    /// Retained for future extensibility; not used to drive model loading.
    pub reranker_model_name: String,
    pub hf_token: String,
    /// Embedding output dimension.  Defaults to 768 (nomic-embed-text-v1.5).
    pub dimension: usize,
    pub disable_watcher: bool,
    pub embedder_pool_size: usize,
    pub api_port: String,
    pub feature_toggles: FeatureToggles,
}

impl Config {
    fn resolve_paths(home: &Path) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf, String)> {
        let data_dir = env_or_default("DATA_DIR", || {
            home.join(".local")
                .join("share")
                .join("vector-mcp-rust")
                .display()
                .to_string()
        });
        let data_dir = PathBuf::from(data_dir);

        let db_path = env_or_default("DB_PATH", || data_dir.join("lancedb").display().to_string());
        let db_path = PathBuf::from(db_path);

        let models_dir = env_or_default("MODELS_DIR", || {
            data_dir.join("models").display().to_string()
        });
        let models_dir = PathBuf::from(models_dir);

        let log_path = env_or_default("LOG_PATH", || {
            data_dir.join("server.log").display().to_string()
        });
        let log_path = PathBuf::from(log_path);

        let project_root = env::var("PROJECT_ROOT").unwrap_or_else(|_| {
            env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| ".".to_string())
        });

        Ok((data_dir, db_path, models_dir, log_path, project_root))
    }

    /// Load configuration from environment variables (and an optional `.env`
    /// file).
    ///
    /// 1. Attempt to load `.env` (ignore if absent).
    /// 2. Resolve paths with the same fallback chain as the Go version.
    /// 3. Ensure required directories exist.
    pub fn load() -> Result<Self> {
        // Best-effort .env loading — missing file is fine.
        let _ = dotenvy::dotenv();

        let home = dirs_home()?;

        let (data_dir, db_path, models_dir, log_path, project_root) =
            Self::resolve_paths(&home)?;

        // Ensure directories exist.
        std::fs::create_dir_all(&db_path)
            .with_context(|| format!("creating db directory: {}", db_path.display()))?;
        std::fs::create_dir_all(&models_dir)
            .with_context(|| format!("creating models directory: {}", models_dir.display()))?;

        let model_name =
            env::var("MODEL_NAME").unwrap_or_else(|_| "nomic-embed-text-v1.5".to_string());

        let reranker_model_name = env::var("RERANKER_MODEL_NAME").unwrap_or_default();

        let disable_watcher = env::var("DISABLE_FILE_WATCHER")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let embedder_pool_size: usize = env::var("EMBEDDER_POOL_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
            .max(1);

        let api_port = env::var("API_PORT").unwrap_or_else(|_| "47821".to_string());

        // Default to 768 — the output dimension of nomic-embed-text-v1.5.
        let dimension = env::var("EMBEDDING_DIM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(768);

        let feature_toggles = FeatureToggles {
            enable_local_llm: env_bool("ENABLE_LOCAL_LLM"),
            enable_live_indexing: env_bool("ENABLE_LIVE_INDEXING"),
        };

        Ok(Self {
            project_root: std::sync::RwLock::new(project_root),
            data_dir,
            db_path,
            models_dir,
            log_path,
            model_name,
            reranker_model_name,
            hf_token: env::var("HF_TOKEN").unwrap_or_default(),
            dimension,
            disable_watcher,
            embedder_pool_size,
            api_port,
            feature_toggles,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dirs_home() -> Result<PathBuf> {
    env::var("HOME")
        .map(PathBuf::from)
        .or_else(|_| Ok(env::temp_dir()))
}

fn env_or_default<F, P>(key: &str, default_fn: F) -> String
where
    F: FnOnce() -> P,
    P: std::fmt::Display,
{
    match env::var(key) {
        Ok(v) if !v.is_empty() => v,
        _ => default_fn().to_string(),
    }
}

fn env_bool(key: &str) -> bool {
    env::var(key)
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false)
}
