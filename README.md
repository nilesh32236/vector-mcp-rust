# vector-mcp-rust 🦀

A high-performance, standalone Rust implementation of the Model Context Protocol (MCP) server for local code search and semantic analysis.

## Features

- **Blazing Fast Indexing**: Semantic code chunking with `tree-sitter` and OS-level file watching (`notify`).
- **Standalone AI**: 100% local ONNX embeddings (`ort`) and GGUF model summarization (`candle`). No external services (like Ollama) required.
- **Hybrid Search**: Advanced search combining vector similarity (1024-dim) and Full-Text Search (Tantivy).
- **Deep Code Analysis**: Metadata extraction of symbols, calls, and importance scoring.
- **Privacy-First**: Everything stays on your local machine.

## Quick Start

### 1. Prerequisites
- Rust (latest stable)
- `libssl-dev` (on Linux)

### 2. Installation
```bash
git clone https://github.com/nilesh32236/vector-mcp-rust.git
cd vector-mcp-rust
cargo build --release
```

### 3. Usage
On the first run, the server will automatically download the necessary models from Hugging Face (~100MB for embeddings, plus optional llm weights).

```bash
cargo run --release
```

The server will perform an initial scan of the current directory (respecting `.gitignore`) and then watch for changes in the background.

## Configuration

Environment variables can be set in a `.env` file:
- `ENABLE_LIVE_INDEXING`: Toggle background watcher (default: `true`).

## SSE Support (Remote/URL Access)

In addition to standard Stdio, `vector-mcp-rust` supports access via HTTP/SSE, matching the Go implementation's behavior.

- **SSE Endpoint**: `http://localhost:47821/sse`
- **Port Configuration**: Controlled via `API_PORT` (default: 47821).

To test via `curl`:
```bash
# Start connection (emulates what an MCP client does)
curl -N http://localhost:47821/sse
```

## Tools Provided

The server registers 22 MCP tools, including:
- `search_codebase`: Hybrid semantic search with Reranking.
- `get_summarized_context`: Local AI summary of search results.
- `reindex_all`: Force refresh of the entire codebase index.
- `analyze_architecture`: High-level summary of code relationships.
- `find_duplicate_code`: Identify semantic duplication across files.
- `check_dependency_health`: Deep import vs package.json/go.mod analysis.
- `find_dead_code`: Identify unused exported symbols.
- `verify_implementation_gap`: Compare docs/feedback with implementation.
- `find_missing_tests`: Map source symbols to missing test coverage.
- `list_api_endpoints`: Discover potential API routes.
- `get_code_history`: Git history for specific files.

## License
MIT
