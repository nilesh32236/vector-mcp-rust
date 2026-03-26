# vector-mcp-rust: User Guide & Verification 🚀

This guide explains how to verify your server and connect it to AI agents like Claude Desktop.

## 1. Manual Verification (Immediate)

### A. Test SSE Connection (URL)
Run the server and then use `curl` to verify the SSE handshake:

```bash
# Terminal 1: Start the server
cargo run --release

# Terminal 2: Test the SSE endpoint
curl -N http://localhost:47821/sse
```

**Expected Result**: You should see an `event: endpoint` with a data string like `/message?session_id=...`. This confirms the HTTP server is alive and session-aware.

### B. Test a Tool Call (via Stdio)
You can manually pipe a JSON-RPC request into the server to test tool execution:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | cargo run --release --quiet
```

**Expected Result**: A JSON response containing the list of 16+ tools.

---

## 2. Connecting to AI Agents

### A. Claude Desktop (Stdio)
To use `vector-mcp-rust` as a tool provider in Claude Desktop, add it to your configuration:

**Location**:
- Linux: `~/.config/Claude/claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Configuration**:
```json
{
  "mcpServers": {
    "vector-mcp": {
      "command": "/bin/bash",
      "args": ["-c", "cd /home/nilesh/Documents/vector-mcp/vector-mcp-rust && cargo run --release --quiet"]
    }
  }
}
```

### B. Agents supporting SSE (URL)
If your agent supports HTTP/SSE (like many custom web-based agents), point it to:
- **URL**: `http://localhost:47821/sse`

---

## 3. Official Verification (MCP Inspector)
The recommended way to test any MCP server is the [MCP Inspector](https://github.com/modelcontextprotocol/inspector).

### Stdio Test:
```bash
npx @modelcontextprotocol/inspector /home/nilesh/Documents/vector-mcp/vector-mcp-rust/target/release/vector-mcp-rust
```

### SSE Test:
```bash
npx @modelcontextprotocol/inspector http://localhost:47821/sse
```

---

## 4. Troubleshooting
- **Build Lock**: If `cargo` hangs, run `pkill -f vector-mcp-rust` to clear stale builds.
- **Port Conflict**: If `47821` is taken, set `API_PORT=5000` in your `.env` file or shell.
- **Models**: The first run will be slow as it downloads the embedding model (~100MB).
