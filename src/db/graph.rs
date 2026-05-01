use crate::db::Record;
use crate::indexer::chunker::{CallEdge, ImplEdge};
use dashmap::DashMap;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EntityNode {
    pub name: String,
    pub node_type: String,
    pub path: String,
    pub docstring: String,
    /// Structural metadata: method:X, field:X keys.
    pub metadata: HashMap<String, String>,
}

pub struct KnowledgeGraph {
    nodes: DashMap<String, EntityNode>,
    /// interface_name → vec of node IDs that implement it.
    impls: DashMap<String, Vec<String>>,
    /// interface_name -> method_name -> signature/meta
    interfaces: DashMap<String, HashMap<String, String>>,
    /// caller_symbol → Vec<CallEdge>
    call_edges: DashMap<String, Vec<CallEdge>>,
    /// trait_name → Vec<struct_name>
    trait_impls: DashMap<String, Vec<String>>,
    /// Secondary index: node name → record ID for O(1) name lookups.
    name_to_id: DashMap<String, String>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            nodes: DashMap::new(),
            impls: DashMap::new(),
            interfaces: DashMap::new(),
            call_edges: DashMap::new(),
            trait_impls: DashMap::new(),
            name_to_id: DashMap::new(),
        }
    }

    /// Add or update a single record in the graph.
    pub fn add_record(&self, r: &Record) {
        let meta = r.metadata_json();
        let name = meta["name"].as_str().unwrap_or("").to_string();
        if name.is_empty() {
            return;
        }
        let node_type = meta["type"].as_str().unwrap_or("").to_string();
        let path = meta["path"].as_str().unwrap_or("").to_string();
        let docstring = meta["docstring"].as_str().unwrap_or("").to_string();

        let struct_meta: HashMap<String, String> = meta["structural_metadata"]
            .as_str()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        let node = EntityNode {
            name: name.clone(),
            node_type: node_type.clone(),
            path,
            docstring,
            metadata: struct_meta.clone(),
        };

        self.nodes.insert(r.id.clone(), node.clone());
        // Maintain secondary name → id index.
        self.name_to_id.insert(name.clone(), r.id.clone());

        // --- Existing interface/impl logic ---
        if node_type == "interface_type" || node_type == "interface" {
            let methods: HashMap<String, String> = struct_meta
                .iter()
                .filter(|(k, _)| k.starts_with("method:"))
                .map(|(k, v)| (k.trim_start_matches("method:").to_string(), v.clone()))
                .collect();

            self.interfaces.insert(name.clone(), methods.clone());

            let mut new_impls = Vec::new();
            for entry in self.nodes.iter() {
                let id = entry.key();
                let other_node = entry.value();
                if (other_node.node_type == "struct_type"
                    || other_node.node_type == "struct"
                    || other_node.node_type == "class")
                    && !methods.is_empty()
                    && methods
                        .keys()
                        .all(|m| other_node.metadata.contains_key(&format!("method:{m}")))
                {
                    new_impls.push(id.clone());
                }
            }
            self.impls.insert(name, new_impls);
        } else if node_type == "struct_type" || node_type == "struct" || node_type == "class" {
            for entry in self.interfaces.iter() {
                let iface_name = entry.key();
                let iface_methods = entry.value();
                if iface_methods.is_empty() {
                    continue;
                }
                let implements = iface_methods
                    .keys()
                    .all(|m| node.metadata.contains_key(&format!("method:{m}")));

                let mut iface_impls = self.impls.entry(iface_name.clone()).or_default();
                if implements {
                    if !iface_impls.contains(&r.id) {
                        iface_impls.push(r.id.clone());
                    }
                } else {
                    iface_impls.retain(|id| id != &r.id);
                }
            }
        }

        // --- New: call_edges from callee_relationships (deduplicated) ---
        let callee_rels: Vec<CallEdge> = meta["callee_relationships"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        for edge in &callee_rels {
            let mut edges = self.call_edges.entry(edge.caller_symbol.clone()).or_default();
            // Deduplicate by (caller_symbol, callee_symbol) pair.
            let already_present = edges
                .iter()
                .any(|e| e.callee_symbol == edge.callee_symbol);
            if !already_present {
                edges.push(edge.clone());
            }
        }

        // --- New: trait_impls from impl_relationships (deduplicated) ---
        let impl_rels: Vec<ImplEdge> = meta["impl_relationships"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        for edge in &impl_rels {
            let mut impls = self.trait_impls.entry(edge.trait_name.clone()).or_default();
            if !impls.contains(&edge.struct_name) {
                impls.push(edge.struct_name.clone());
            }
        }
    }

    /// Rebuild the graph from a fresh set of DB records.
    pub fn populate(&self, records: &[Record]) {
        self.nodes.clear();
        self.impls.clear();
        self.interfaces.clear();
        self.call_edges.clear();
        self.trait_impls.clear();
        self.name_to_id.clear();

        for r in records {
            self.add_record(r);
        }
    }

    // -----------------------------------------------------------------------
    // Existing query methods
    // -----------------------------------------------------------------------

    pub fn get_implementations(&self, interface_name: &str) -> Vec<EntityNode> {
        self.impls
            .get(interface_name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.nodes.get(id).map(|n| n.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn find_usage(&self, field_name: &str) -> Vec<EntityNode> {
        let key = format!("field:{field_name}");
        self.nodes
            .iter()
            .filter(|n| n.value().metadata.contains_key(&key))
            .map(|n| n.value().clone())
            .collect()
    }

    pub fn search_by_name(&self, name: &str) -> Vec<EntityNode> {
        let lower = name.to_lowercase();
        self.nodes
            .iter()
            .filter(|n| n.value().name.to_lowercase().contains(&lower))
            .map(|n| n.value().clone())
            .collect()
    }

    // -----------------------------------------------------------------------
    // New GraphRAG query methods
    // -----------------------------------------------------------------------

    /// Returns all nodes whose symbol calls `callee_symbol`.
    pub fn get_callers(&self, callee_symbol: &str) -> Vec<EntityNode> {
        let mut callers = Vec::new();
        for entry in self.call_edges.iter() {
            let caller_sym = entry.key();
            let edges = entry.value();
            if edges.iter().any(|e| e.callee_symbol == callee_symbol) {
                // O(1) lookup via name_to_id index.
                if let Some(id) = self.name_to_id.get(caller_sym) {
                    if let Some(node) = self.nodes.get(id.value()) {
                        callers.push(node.clone());
                    }
                }
            }
        }
        callers
    }

    /// Returns all nodes called by `caller_symbol`.
    pub fn get_callees(&self, caller_symbol: &str) -> Vec<EntityNode> {
        self.call_edges
            .get(caller_symbol)
            .map(|edges| {
                edges
                    .iter()
                    .filter_map(|e| {
                        // O(1) lookup via name_to_id index.
                        self.name_to_id
                            .get(&e.callee_symbol)
                            .and_then(|id| self.nodes.get(id.value()).map(|n| n.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns all structs/classes implementing `trait_name`.
    pub fn get_impl_chain(&self, trait_name: &str) -> Vec<EntityNode> {
        self.trait_impls
            .get(trait_name)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|struct_name| {
                        // O(1) lookup via name_to_id index.
                        self.name_to_id
                            .get(struct_name)
                            .and_then(|id| self.nodes.get(id.value()).map(|n| n.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    // -----------------------------------------------------------------------
    // Stats helpers
    // -----------------------------------------------------------------------

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn call_edge_count(&self) -> usize {
        self.call_edges.iter().map(|e| e.value().len()).sum()
    }

    pub fn impl_edge_count(&self) -> usize {
        self.trait_impls.iter().map(|e| e.value().len()).sum()
    }
}
