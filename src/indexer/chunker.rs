//! # AST Semantic Chunker
//!
//! This module uses [Tree-sitter](https://tree-sitter.github.io/) to parse
//! source files into semantically meaningful chunks (functions, classes,
//! structs, interfaces, methods) rather than slicing at arbitrary byte
//! boundaries.
//!
//! ## Rust vs Go Lifetime Notes (for the Go developer)
//!
//! In the Go codebase (`chunker.go`) you'll see patterns like:
//!
//! ```go
//! parser := sitter.NewParser()
//! defer parser.Close()
//! ```
//!
//! This is necessary because Go's garbage collector doesn't know about the
//! C-allocated memory that Tree-sitter owns internally.  Without `Close()`
//! (or `defer`d cleanup), that C memory would leak.
//!
//! In Rust, every Tree-sitter type (`Parser`, `Tree`, `Query`, `QueryCursor`)
//! implements the **`Drop` trait**.  When a value goes out of scope, Rust
//! *automatically* calls `Drop::drop()`, which in turn calls the
//! corresponding C destructor (`ts_parser_delete`, `ts_tree_delete`, etc.).
//! This means:
//!
//! - **No `defer` required.**  The value is freed when the enclosing block
//!   ends (or when the `Vec` / `Option` owning it is dropped).
//! - **No `Close()` methods.**  There is nothing to call manually.
//! - **Double-free is impossible** because Rust's ownership system guarantees
//!   that each value has exactly one owner.
//!
//! As a result, all Tree-sitter resources in this file are managed purely
//! through scope: the parser, tree, query, and cursor are all dropped at the
//! end of the function that creates them.

use std::collections::HashSet;

use anyhow::{Context, Result};
use tree_sitter::{Language, Node, Parser, Query, QueryCursor, StreamingIterator};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A semantically meaningful slice of a source file.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The raw source text of this chunk.
    pub content: String,
    /// Human-readable summary  (e.g. "File: foo.ts. Entity: parse_file. …").
    pub contextual_string: String,
    /// Symbol names exported by this chunk (function/class/struct names).
    pub symbols: Vec<String>,
    /// Import paths or module references discovered in the file.
    pub relationships: Vec<String>,
    /// AST node type (e.g. "function_declaration", "class_declaration").
    pub node_type: String,
    /// Function/method calls made inside this chunk.
    pub calls: Vec<String>,
    /// Heuristic importance score (higher = more relevant).
    pub function_score: f32,
    /// 1-based start line number.
    pub start_line: usize,
    /// 1-based end line number.
    pub end_line: usize,
}

// ---------------------------------------------------------------------------
// Language / query resolution
// ---------------------------------------------------------------------------

/// Resolved language + its S-expression queries.
struct LangSpec {
    language: Language,
    queries: &'static [&'static str],
}

/// Map a file extension to a Tree-sitter language and query set.
fn resolve_language(ext: &str) -> Option<LangSpec> {
    let (language, queries): (Language, &[&str]) = match ext {
        ".go" => (tree_sitter_go::LANGUAGE.into(), GO_QUERIES),
        ".js" | ".jsx" => (tree_sitter_javascript::LANGUAGE.into(), JS_TS_QUERIES),
        ".ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            JS_TS_QUERIES,
        ),
        ".tsx" => (tree_sitter_typescript::LANGUAGE_TSX.into(), JS_TS_QUERIES),
        ".php" => (tree_sitter_php::LANGUAGE_PHP.into(), PHP_QUERIES),
        ".rs" => (tree_sitter_rust::LANGUAGE.into(), RUST_QUERIES),
        ".py" => (tree_sitter_python::LANGUAGE.into(), PYTHON_QUERIES),
        _ => return None,
    };

    Some(LangSpec { language, queries })
}

// ---------------------------------------------------------------------------
// S-expression queries (mirrored from Go `chunker.go`)
// ---------------------------------------------------------------------------

const GO_QUERIES: &[&str] = &[
    "(function_declaration name: (identifier) @name) @entity",
    "(method_declaration name: (field_identifier) @name) @entity",
    "(type_declaration (type_spec name: (type_identifier) @name type: (struct_type))) @entity",
    "(type_declaration (type_spec name: (type_identifier) @name type: (interface_type))) @entity",
];

const JS_TS_QUERIES: &[&str] = &[
    "(export_statement declaration: (class_declaration name: (type_identifier) @name)) @entity",
    "(class_declaration name: (type_identifier) @name) @entity",
    "(export_statement declaration: (function_declaration name: (identifier) @name)) @entity",
    "(function_declaration name: (identifier) @name) @entity",
    "(export_statement (interface_declaration name: (type_identifier) @name)) @entity",
    "(interface_declaration name: (type_identifier) @name) @entity",
    "(method_definition name: (property_identifier) @name) @entity",
    r#"(lexical_declaration (variable_declarator name: (identifier) @name value: [(arrow_function) (function)])) @entity"#,
    r#"(export_statement declaration: (lexical_declaration (variable_declarator name: (identifier) @name value: [(arrow_function) (function)]))) @entity"#,
];

const PHP_QUERIES: &[&str] = &[
    "(class_declaration name: (name) @name) @entity",
    "(method_declaration name: (name) @name) @entity",
    "(function_declaration name: (name) @name) @entity",
    "(interface_declaration name: (name) @name) @entity",
];

const RUST_QUERIES: &[&str] = &[
    "(function_item name: (identifier) @name) @entity",
    "(struct_item name: (type_identifier) @name) @entity",
    "(enum_item name: (type_identifier) @name) @entity",
    "(trait_item name: (type_identifier) @name) @entity",
    "(impl_item type: (type_identifier) @name) @entity",
];

/// Python S-expression queries — mirrors Go's `chunker.go` PYTHON_QUERIES block.
///
/// Captures top-level and class-level function definitions, class declarations,
/// and decorated functions (e.g. `@staticmethod`, `@property`).
const PYTHON_QUERIES: &[&str] = &[
    // Top-level function definitions
    "(function_definition name: (identifier) @name) @entity",
    // Class definitions
    "(class_definition name: (identifier) @name) @entity",
    // Decorated definitions (e.g. @staticmethod, @classmethod, @property)
    "(decorated_definition definition: (function_definition name: (identifier) @name)) @entity",
    "(decorated_definition definition: (class_definition name: (identifier) @name)) @entity",
];

// ---------------------------------------------------------------------------
// Core public function
// ---------------------------------------------------------------------------

/// Parse `content` using Tree-sitter for the given file `extension` and
/// return semantically meaningful chunks.
pub fn parse_file(content: &str, file_path: &str, extension: &str) -> Result<Vec<Chunk>> {
    if extension == ".pdf" {
        return parse_pdf(content.as_bytes(), file_path);
    }

    let relationships = parse_relationships(content, extension);

    let spec = match resolve_language(extension) {
        Some(s) => s,
        None => {
            let mut chunks = fast_chunk(content, file_path);
            for c in &mut chunks {
                c.relationships.clone_from(&relationships);
                c.contextual_string = format!(
                    "File: {file_path}. Entity: Global. Type: {}. Relationships: {:?}. Code:\n{}",
                    c.node_type, c.relationships, c.content
                );
            }
            return Ok(chunks);
        }
    };

    let chunks = match tree_sitter_chunk(content, file_path, &spec) {
        Ok(c) if !c.is_empty() => c,
        _ => fast_chunk(content, file_path),
    };

    // Stamp relationships + contextual string onto every chunk.
    let chunks = chunks
        .into_iter()
        .map(|mut c| {
            c.relationships.clone_from(&relationships);
            let scope = c.symbols.first().map_or("Global", String::as_str);
            let calls_str = if c.calls.is_empty() {
                "None".to_owned()
            } else {
                c.calls.join(", ")
            };
            c.contextual_string = format!(
                "File: {file_path}. Entity: {scope}. Type: {}. Calls: {calls_str}. Code:\n{}",
                c.node_type, c.content
            );
            c
        })
        .collect();

    Ok(chunks)
}

/// Extract text from PDF bytes and return chunks.
pub fn parse_pdf(bytes: &[u8], file_path: &str) -> Result<Vec<Chunk>> {
    let text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| anyhow::anyhow!("PDF extraction failed: {}", e))?;
    Ok(fast_chunk(&text, file_path))
}

// ---------------------------------------------------------------------------
// Tree-sitter chunking
// ---------------------------------------------------------------------------

fn extract_raw_chunks(source: &[u8], spec: &LangSpec, tree: &tree_sitter::Tree) -> Vec<Chunk> {
    let root = tree.root_node();
    let mut raw_chunks: Vec<Chunk> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();

    for query_str in spec.queries {
        let query = match Query::new(&spec.language, query_str) {
            Ok(q) => q,
            Err(_) => continue, // skip invalid queries gracefully
        };

        let entity_idx = query.capture_names().iter().position(|n| *n == "entity");
        let name_idx = query
            .capture_names()
            .iter()
            .position(|n| *n == "name" || *n == "hook_name");

        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, root, source);

        while let Some(m) = matches.next() {
            let entity_node: Option<Node> = entity_idx.and_then(|idx| {
                m.captures
                    .iter()
                    .find(|c| c.index as usize == idx)
                    .map(|c| c.node)
            });
            let name_node: Option<Node> = name_idx.and_then(|idx| {
                m.captures
                    .iter()
                    .find(|c| c.index as usize == idx)
                    .map(|c| c.node)
            });

            let entity_node: Node = match entity_node {
                Some(n) => n,
                None => continue,
            };

            let start_byte = entity_node.start_byte();
            let end_byte = entity_node.end_byte();
            let key = (start_byte, end_byte);

            if !seen.insert(key) {
                continue;
            }

            let symbol_name = name_node
                .and_then(|n: Node| n.utf8_text(source).ok())
                .unwrap_or("Unknown")
                .to_owned();

            let chunk_text = entity_node.utf8_text(source).unwrap_or("").to_owned();

            let calls = extract_calls(entity_node, source);
            let score = calculate_score(entity_node, &calls);

            raw_chunks.push(Chunk {
                content: chunk_text,
                contextual_string: String::new(), // filled later
                symbols: vec![symbol_name],
                relationships: Vec::new(), // filled later
                node_type: entity_node.kind().to_owned(),
                calls,
                function_score: score,
                start_line: entity_node.start_position().row + 1,
                end_line: entity_node.end_position().row + 1,
            });
        }
    }
    raw_chunks
}

fn deduplicate_chunks(raw_chunks: Vec<Chunk>) -> Vec<Chunk> {
    raw_chunks
        .iter()
        .enumerate()
        .filter(|&(i, c1)| {
            !raw_chunks.iter().enumerate().any(|(j, c2)| {
                i != j
                    && c1.start_line >= c2.start_line
                    && c1.end_line <= c2.end_line
                    && c1.node_type == c2.node_type
                    && (c1.end_line - c1.start_line) < (c2.end_line - c2.start_line)
            })
        })
        .map(|(_, c)| c.clone())
        .collect()
}

fn tree_sitter_chunk(content: &str, _file_path: &str, spec: &LangSpec) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&spec.language)
        .context("setting Tree-sitter language")?;

    let tree = parser
        .parse(content, None)
        .context("Tree-sitter parse returned None")?;

    let source = content.as_bytes();
    let raw_chunks = extract_raw_chunks(source, spec, &tree);

    // Deduplicate: remove smaller chunks fully contained in a larger chunk of
    // the same type (mirrors Go's redundancy filter).
    let filtered = deduplicate_chunks(raw_chunks);

    Ok(filtered)
}

// ---------------------------------------------------------------------------
// Call extraction (mirrors Go's `extractCallsGeneric`)
// ---------------------------------------------------------------------------

fn extract_calls(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut unique: HashSet<String> = HashSet::new();
    walk_calls(node, source, &mut unique);
    let mut calls: Vec<String> = unique.into_iter().collect();
    calls.sort();
    calls
}

fn walk_calls(node: tree_sitter::Node, source: &[u8], out: &mut HashSet<String>) {
    let kind = node.kind();
    if kind == "call_expression" || kind == "function_call_expression" {
        let child_count = node.child_count();
        for i in 0..child_count {
            if let Some(child) = node.child(i) {
                let ct = child.kind();
                if ct == "identifier" || ct == "property_identifier" || ct == "name" {
                    if let Ok(text) = child.utf8_text(source) {
                        out.insert(text.to_owned());
                    }
                } else if (ct == "selector_expression"
                    || ct == "member_expression"
                    || ct == "field_expression")
                    && child.child_count() > 0
                    && let Some(last) = child.child(child.child_count() - 1)
                {
                    let lt = last.kind();
                    if (lt == "field_identifier" || lt == "property_identifier")
                        && let Ok(text) = last.utf8_text(source)
                    {
                        out.insert(text.to_owned());
                    }
                }
            }
        }
    }

    let child_count = node.child_count();
    for i in 0..child_count {
        if let Some(child) = node.child(i) {
            walk_calls(child, source, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Heuristic scoring (mirrors Go's `calculateScoreGeneric`)
// ---------------------------------------------------------------------------

fn calculate_score(node: tree_sitter::Node, calls: &[String]) -> f32 {
    let mut score: f32 = 1.0;
    let lines = (node.end_position().row - node.start_position().row + 1) as i32;

    if lines < 3 {
        score -= 0.3;
    } else if lines > 10 {
        score += 0.2;
    }

    score += calls.len() as f32 * 0.1;
    score
}

// ---------------------------------------------------------------------------
// Relationship parsing (mirrors Go's `parseRelationships`)
// ---------------------------------------------------------------------------

use std::sync::LazyLock;

static JS_PATH_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r#"(?:import|from|require)\s*\(?\s*['"]([^'"]+)['"]"#).unwrap()
});
static JS_NAMED_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"import\s*\{([^}]+)\}").unwrap());

static GO_SINGLE_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r#"import\s+(?:[a-zA-Z0-9_.]+\s+)?["']([^"']+)["']"#).unwrap()
});
static GO_BLOCK_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"import\s+\(([\s\S]*?)\)").unwrap());
static GO_INNER_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r#"["']([^"']+)["']"#).unwrap());

static PHP_REQ_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r#"(?:require|require_once|include|include_once)\s*\(?\s*['"]([^'"]+)['"]"#)
        .unwrap()
});
static PHP_USE_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"use\s+([^;]+);").unwrap());

static RUST_USE_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"use\s+([^;]+);").unwrap());

fn parse_js_relationships(text: &str, relations: &mut Vec<String>) {
    // import/from/require paths
    for cap in JS_PATH_RE.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            relations.push(m.as_str().to_owned());
        }
    }
    // named imports
    for cap in JS_NAMED_RE.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            for name in m.as_str().split(',') {
                let cleaned = name.split(" as ").next().unwrap_or("").trim();
                if !cleaned.is_empty() {
                    relations.push(cleaned.to_owned());
                }
            }
        }
    }
}

fn parse_go_relationships(text: &str, relations: &mut Vec<String>) {
    // single import
    for cap in GO_SINGLE_RE.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            relations.push(m.as_str().to_owned());
        }
    }
    // block import
    for cap in GO_BLOCK_RE.captures_iter(text) {
        if let Some(block) = cap.get(1) {
            for ic in GO_INNER_RE.captures_iter(block.as_str()) {
                if let Some(m) = ic.get(1) {
                    relations.push(m.as_str().to_owned());
                }
            }
        }
    }
}

fn parse_php_relationships(text: &str, relations: &mut Vec<String>) {
    for cap in PHP_REQ_RE.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            relations.push(m.as_str().to_owned());
        }
    }
    for cap in PHP_USE_RE.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            for part in m.as_str().split(',') {
                let cleaned = part.split(" as ").next().unwrap_or("").trim();
                if !cleaned.is_empty() {
                    relations.push(cleaned.to_owned());
                }
            }
        }
    }
}

fn parse_rust_relationships(text: &str, relations: &mut Vec<String>) {
    for cap in RUST_USE_RE.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            relations.push(m.as_str().trim().to_owned());
        }
    }
}

fn parse_relationships(text: &str, ext: &str) -> Vec<String> {
    let mut relations: Vec<String> = Vec::new();

    match ext {
        ".ts" | ".tsx" | ".js" | ".jsx" => parse_js_relationships(text, &mut relations),
        ".go" => parse_go_relationships(text, &mut relations),
        ".php" => parse_php_relationships(text, &mut relations),
        ".rs" => parse_rust_relationships(text, &mut relations),
        _ => {}
    }

    // Deduplicate while preserving order.
    let mut seen = HashSet::new();
    relations.retain(|r| seen.insert(r.clone()));
    relations
}

// ---------------------------------------------------------------------------
// Fallback: sliding-window chunker (mirrors Go's `fastChunk`)
// ---------------------------------------------------------------------------

const CHUNK_SIZE: usize = 3000;
const OVERLAP: usize = 500;

fn fast_chunk(text: &str, file_path: &str) -> Vec<Chunk> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let end = (i + CHUNK_SIZE).min(chars.len());
        let content: String = chars[i..end].iter().collect();
        let prefix: String = chars[..i].iter().collect();
        let start_line = prefix.matches('\n').count() + 1;
        let end_line = start_line + content.matches('\n').count();

        chunks.push(Chunk {
            content: content.clone(),
            contextual_string: format!(
                "File: {file_path}. Entity: Global. Type: chunk. Calls: None. Code:\n{content}"
            ),
            symbols: Vec::new(),
            relationships: Vec::new(),
            node_type: "chunk".to_owned(),
            calls: Vec::new(),
            function_score: 0.0,
            start_line,
            end_line,
        });

        if end == chars.len() {
            break;
        }

        i += CHUNK_SIZE - OVERLAP;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_js_relationships() {
        let code = r#"
            import { foo } from "./foo";
            import * as bar from "bar";
            const baz = require('baz');
        "#;
        let mut relations = Vec::new();
        parse_js_relationships(code, &mut relations);
        assert!(relations.contains(&"./foo".to_string()));
        assert!(relations.contains(&"foo".to_string()));
        assert!(relations.contains(&"bar".to_string()));
        assert!(relations.contains(&"baz".to_string()));
    }

    #[test]
    fn test_parse_go_relationships() {
        let code = r#"
            import "fmt"
            import (
                "os"
                "path/filepath"
            )
        "#;
        let mut relations = Vec::new();
        parse_go_relationships(code, &mut relations);
        assert!(relations.contains(&"fmt".to_string()));
        assert!(relations.contains(&"os".to_string()));
        assert!(relations.contains(&"path/filepath".to_string()));
    }

    #[test]
    fn test_parse_rust_relationships() {
        let code = r#"
            use std::collections::HashMap;
            use crate::db::Store;
        "#;
        let mut relations = Vec::new();
        parse_rust_relationships(code, &mut relations);
        assert!(relations.contains(&"std::collections::HashMap".to_string()));
        assert!(relations.contains(&"crate::db::Store".to_string()));
    }

    #[test]
    fn test_parse_relationships_dispatch() {
        let code = "import 'vue';";
        let rels = parse_relationships(code, ".js");
        assert!(rels.contains(&"vue".to_string()));

        let go_code = "import \"net/http\"";
        let go_rels = parse_relationships(go_code, ".go");
        assert!(go_rels.contains(&"net/http".to_string()));
    }
}
