#[path = "../api/mod.rs"] pub mod api;
#[path = "../config/mod.rs"] pub mod config;
#[path = "../daemon/mod.rs"] pub mod daemon;
#[path = "../db/mod.rs"] pub mod db;
#[path = "../indexer/mod.rs"] pub mod indexer;
#[path = "../llm/mod.rs"] pub mod llm;
#[path = "../lsp/mod.rs"] pub mod lsp;
#[path = "../mcp/mod.rs"] pub mod mcp;
#[path = "../mutation/mod.rs"] pub mod mutation;
#[path = "../security/mod.rs"] pub mod security;
#[path = "../benchmark.rs"] pub mod benchmark;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    benchmark::run(args)
}
