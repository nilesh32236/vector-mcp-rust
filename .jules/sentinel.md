## 2025-04-06 - [Path Traversal in MCP File Handlers]
**Vulnerability:** User-supplied file paths in `handle_get_codebase_skeleton` and `handle_check_dependency_health` were concatenated with the project root or allowed as absolute paths, enabling path traversal beyond the project directory.
**Learning:** Even when the project root is defined, directly joining user input or conditionally allowing absolute paths can bypass directory restrictions.
**Prevention:** Always use `server.path_guard.validate(&path, PathOp::Read)` (or `PathOp::Create`) to resolve and validate user-supplied file paths against the base directory.
