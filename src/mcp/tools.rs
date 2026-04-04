//! MCP tool registry — Fat Tool pattern.
//!
//! Exposes exactly **5 consolidated tools** to the LLM client, matching the Go v2 design.
//! Each tool accepts an `action` parameter that dispatches to the appropriate sub-handler.
//!
//! Design rule (from Go AGENTS.md): new capabilities are added as `action` enum variants
//! inside existing tools, **never** as new top-level tools. This prevents LLM context
//! exhaustion and tool-selection confusion.

use serde_json::json;

use super::protocol::{ToolInfo, ToolInputSchema};

// ---------------------------------------------------------------------------
// Public registry
// ---------------------------------------------------------------------------

/// Returns the 5 Fat Tools exposed to the LLM.
pub fn tool_definitions() -> Vec<ToolInfo> {
    vec![
        tool_search_workspace(),
        tool_workspace_manager(),
        tool_analyze_code(),
        tool_modify_workspace(),
        tool_lsp_query(),
    ]
}

// ---------------------------------------------------------------------------
// Fat Tool definitions
// ---------------------------------------------------------------------------

/// Unified search engine: semantic vector search, exact regex grep, knowledge-graph
/// traversal, and index status — all behind a single `action` discriminator.
fn tool_search_workspace() -> ToolInfo {
    ToolInfo {
        name: "search_workspace".into(),
        description: concat!(
            "Unified codebase search. ",
            "action='vector': semantic similarity search. ",
            "action='regex': exact text / regex grep across files. ",
            "action='graph': knowledge-graph traversal (interface implementations, symbol usage). ",
            "action='index_status': check background indexing progress."
        )
        .into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "action": {
                    "type": "string",
                    "description": "vector | regex | graph | index_status"
                },
                "query": {
                    "type": "string",
                    "description": "Search query, symbol name, or regex pattern"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum results to return (default 10)"
                },
                "path": {
                    "type": "string",
                    "description": "Optional: restrict search to files whose path contains this string"
                }
            }),
            required: vec![],
        },
    }
}

/// Project lifecycle management: switch active project root, trigger re-indexing,
/// retrieve system diagnostics, and manage stored context / knowledge items.
fn tool_workspace_manager() -> ToolInfo {
    ToolInfo {
        name: "workspace_manager".into(),
        description: concat!(
            "Project lifecycle management. ",
            "action='set_project_root': switch the active workspace. ",
            "action='trigger_index': start a full background re-index. ",
            "action='get_indexing_diagnostics': detailed health and progress report. ",
            "action='store_context': persist architectural decisions or rules. ",
            "action='delete_context': remove stored context entries."
        )
        .into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "action": {
                    "type": "string",
                    "description": "set_project_root | trigger_index | get_indexing_diagnostics | store_context | delete_context"
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path for set_project_root / trigger_index"
                },
                "text": {
                    "type": "string",
                    "description": "Context text for store_context"
                },
                "target_path": {
                    "type": "string",
                    "description": "File path or 'ALL' for delete_context"
                }
            }),
            required: vec![],
        },
    }
}

/// Codebase diagnostic suite: AST skeleton, dead-code detection, semantic duplicate
/// detection, dependency health validation, and package distillation.
fn tool_analyze_code() -> ToolInfo {
    ToolInfo {
        name: "analyze_code".into(),
        description: concat!(
            "Advanced codebase diagnostics. ",
            "action='ast_skeleton': directory tree with file/folder structure. ",
            "action='dead_code': find exported symbols never imported or called. ",
            "action='duplicate_code': detect semantically similar code blocks. ",
            "action='dependencies': validate package.json / go.mod / requirements.txt. ",
            "action='distill_package': summarise a package and re-index with 2x priority."
        )
        .into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "action": {
                    "type": "string",
                    "description": "ast_skeleton | dead_code | duplicate_code | dependencies | distill_package"
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory or file path to analyse (defaults to project root)"
                },
                "is_library": {
                    "type": "boolean",
                    "description": "dead_code only: set true for library packages to skip public exports"
                }
            }),
            required: vec![],
        },
    }
}

/// Safe, guarded workspace mutations with optional LSP-verified patch integrity checking.
fn tool_modify_workspace() -> ToolInfo {
    ToolInfo {
        name: "modify_workspace".into(),
        description: concat!(
            "Safe file mutation tools. ",
            "action='apply_patch': search-and-replace with LSP diagnostic verification. ",
            "action='create_file': create a new file with content. ",
            "action='run_linter': run a formatter (gofmt | rustfmt | prettier). ",
            "action='verify_patch': dry-run — check if a patch is applicable without writing."
        )
        .into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "action": {
                    "type": "string",
                    "description": "apply_patch | create_file | run_linter | verify_patch"
                },
                "path":    { "type": "string", "description": "Target file path" },
                "content": { "type": "string", "description": "File content for create_file" },
                "search":  { "type": "string", "description": "Exact text block to find (apply_patch / verify_patch)" },
                "replace": { "type": "string", "description": "Replacement text (apply_patch)" },
                "tool":    { "type": "string", "description": "Formatter name: gofmt | rustfmt | prettier" }
            }),
            required: vec![],
        },
    }
}

/// High-precision Language Server Protocol queries: symbol definitions, all references,
/// type hierarchies, and blast-radius impact analysis.
fn tool_lsp_query() -> ToolInfo {
    ToolInfo {
        name: "lsp_query".into(),
        description: concat!(
            "LSP-powered symbol intelligence. ",
            "action='definition': jump to the source definition of a symbol. ",
            "action='references': find all usages across the workspace. ",
            "action='type_hierarchy': explore supertypes and subtypes. ",
            "action='impact_analysis': blast-radius — how many files are affected by changing this symbol."
        ).into(),
        input_schema: ToolInputSchema {
            schema_type: "object",
            properties: json!({
                "action": {
                    "type": "string",
                    "description": "definition | references | type_hierarchy | impact_analysis"
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file containing the symbol"
                },
                "line": {
                    "type": "number",
                    "description": "0-indexed line number of the symbol"
                },
                "character": {
                    "type": "number",
                    "description": "0-indexed character offset of the symbol"
                }
            }),
            required: vec!["action".into(), "path".into()],
        },
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn empty_schema() -> ToolInputSchema {
    ToolInputSchema {
        schema_type: "object",
        properties: json!({}),
        required: vec![],
    }
}
