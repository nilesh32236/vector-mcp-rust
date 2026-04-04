# vector-mcp-rust: User Guide & Verification 🚀

This guide explains how to build your server, configure it, and connect it to AI agents like Claude Desktop.

## 1. Building the Server

The server is written in Rust and requires `cargo`. It compiles to a single, statically-linked binary optimized for CPU.

```bash
cd vector-mcp-rust
cargo build --release
```

The compiled binary will be located at `target/release/vector-mcp-rust`.

## 2. Configuration

Create a `.env` file in your project root or set the environment variables globally:

```env
# Path to store the LanceDB vector database
MCP_DB_PATH=~/.local/share/vector-mcp-rust/db

# Port for the HTTP API and SSE connection
API_PORT=47821

# Enable/Disable features
ENABLE_LIVE_INDEXING=true
ENABLE_LOCAL_LLM=true

# (Optional) API key for Gemini if you want to use cloud distillation
# instead of the local GGUF model for large package summaries.
GEMINI_API_KEY=your_key_here
```

## 3. Running as a Standalone Server

You can run the server directly in your terminal. It will default to the current directory as the project root.

```bash
./target/release/vector-mcp-rust
```

When you start the server for the first time, it will automatically download the embedding model (ONNX) and the summarization model (GGUF) to your local cache. **This may take a few minutes depending on your internet connection.**

## 4. Connecting to Claude Desktop

To use this server with Claude Desktop, you need to configure it as an MCP tool provider.

Open your Claude Desktop config file:
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Add the following configuration:

```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "/absolute/path/to/your/vector-mcp-rust/target/release/vector-mcp-rust",
      "env": {
        "API_PORT": "47821",
        "RUST_LOG": "info"
      }
    }
  }
}
```

Restart Claude Desktop.

## 5. Troubleshooting

-   **"Method not found" / Timeout**: Ensure the binary path in `claude_desktop_config.json` is absolute and correct.
-   **High Memory Usage**: The server spawns LSP child processes (like `tsserver`) to verify patches. These will automatically shut down after 10 minutes of inactivity.
-   **Port Conflicts**: If port `47821` is in use, change `API_PORT` in your `.env` or Claude config.
-   **Logs**: Check `~/.local/share/vector-mcp-rust/mcp.log` for detailed JSON-formatted logs without interrupting the MCP JSON-RPC stream.
