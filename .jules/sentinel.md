## 2026-04-06 - [Path Traversal in MCP File Handlers]

**Vulnerability:** User-supplied file paths in `handle_get_codebase_skeleton` and `handle_check_dependency_health` were concatenated with the project root or allowed as absolute paths, enabling path traversal beyond the project directory.
**Learning:** Even when the project root is defined, directly joining user input or conditionally allowing absolute paths can bypass directory restrictions.
**Prevention:** Always use `server.path_guard.validate(&path, PathOp::Read)` (or `PathOp::Create`) to resolve and validate user-supplied file paths against the base directory.

## 2026-04-09 - [Path Traversal in MCP Handlers]
**Vulnerability:** Found Path Traversal / Arbitrary File Read vulnerabilities in `handle_set_project_root`, `handle_trigger_project_index`, and `handle_lsp_query` within `src/mcp/handlers.rs`. User-supplied paths were not being validated against the configured path guard, potentially allowing attackers to read files outside the designated workspace.
**Learning:** Even though a `PathGuard` utility exists, it must be explicitly invoked for all user-supplied paths acting as endpoints. These parameters act as trust boundaries that need validation before interacting with the file system.
**Prevention:** Always wrap external user-supplied path variables with `server.path_guard.validate(&path, PathOp::Read)` (or `PathOp::Create`) to ensure safe resolution and validation boundaries before reading, parsing, or initiating processes within the filesystem.

## 2025-04-13 - [Sentinel: Fix Argument Injection Risks in External Tool Commands]
**Vulnerability:** External formatters (gofmt, rustfmt, prettier) and git were called with paths given as standard string arguments directly without ending the options list. If a file named `-e` was processed, it might be evaluated as an option by the external command, leading to argument injection. Additionally, when invalid UTF-8 paths are passed, `.to_str().unwrap_or("")` creates an empty string which is dangerous.
**Learning:** Argument injection in external tool executions using `tokio::process::Command` happens when user-supplied file names or data might start with hyphens. Moreover, `OsStr` should always be passed to `Command::arg` rather than strings to safely handle paths and prevent silent fallbacks to empty strings.
**Prevention:** Always use `--` to indicate the end of options before passing file paths to CLI tools. Further, always pass `.as_os_str()` to external commands instead of using string conversions.
