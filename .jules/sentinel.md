## 2024-05-15 - Path Traversal in MCP Handlers
**Vulnerability:** The `handle_get_code_history` MCP handler directly executed the `git log` command using a raw user-supplied `file_path` without validation.
**Learning:** Even if the command uses `--` to prevent argument injection, an attacker could still supply paths like `../../../etc/passwd` to traverse outside the intended workspace directory and read sensitive repository history.
**Prevention:** Always validate all user-supplied paths using the existing `server.path_guard.validate(&path)` utility before passing them to file operations or system commands.
