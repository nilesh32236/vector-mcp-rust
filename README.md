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

The server registers 16+ MCP tools, including:
- `search_codebase`: Hybrid semantic search.
- `get_definition`: Find where a symbol is defined.
- `analyze_architecture`: High-level summary of code relationships.
- `find_duplicate_code`: Identify semantic duplication across files.

## License
MIT
