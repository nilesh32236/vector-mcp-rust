## 2024-05-15 - Path Traversal in MCP Handlers
**Vulnerability:** The `handle_get_code_history` MCP handler directly executed the `git log` command using a raw user-supplied `file_path` without validation.
**Learning:** Even if the command uses `--` to prevent argument injection, an attacker could still supply paths like `../../../etc/passwd` to traverse outside the intended workspace directory and read sensitive repository history.
**Prevention:** Always validate all user-supplied paths using the existing `server.path_guard.validate(&path)` utility before passing them to file operations or system commands.

## $(date +%Y-%m-%d) - [Inconsistent PathGuard Usage Leading to Path Traversal]
**Vulnerability:** Path Traversal vulnerabilities in `src/mcp/handlers.rs` (specifically `handle_get_codebase_skeleton` and `handle_check_dependency_health`) where MCP tool arguments like `target_path` and `directory_path` were joined directly with `project_root` or used as absolute paths without validation.
**Learning:** The codebase has a robust internal utility `server.path_guard` (`src/security/pathguard.rs`) designed to prevent directory traversal and symlink escapes. However, it was not consistently applied across all MCP handlers, exposing some handlers to out-of-bounds file access risks.
**Prevention:** Always validate user-supplied file paths using `server.path_guard.validate(&path, crate::security::pathguard::PathOp::Read)` (or `PathOp::Create`) in every MCP handler before using the path. Ensure all subsequent logic relies on the returned canonicalized path rather than the raw user input.
