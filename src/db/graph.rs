#![allow(dead_code)]
//! In-memory knowledge graph — mirrors Go's db/graph.go.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::db::Record;

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
    nodes: RwLock<HashMap<String, EntityNode>>,
    /// interface_name → vec of node IDs that implement it.
    impls: RwLock<HashMap<String, Vec<String>>>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            impls: RwLock::new(HashMap::new()),
        }
    }

    /// Rebuild the graph from a fresh set of DB records.
    pub fn populate(&self, records: &[Record]) {
        let mut nodes: HashMap<String, EntityNode> = HashMap::new();
        let mut interfaces: HashMap<String, HashMap<String, String>> = HashMap::new();

        // First pass — collect nodes.
        for r in records {
            let meta = r.metadata_json();
            let name = meta["name"].as_str().unwrap_or("").to_string();
            if name.is_empty() {
                continue;
            }
            let node_type = meta["type"].as_str().unwrap_or("").to_string();
            let path = meta["path"].as_str().unwrap_or("").to_string();
            let docstring = meta["docstring"].as_str().unwrap_or("").to_string();

            // Parse structural_metadata JSON blob.
            let struct_meta: HashMap<String, String> = meta["structural_metadata"]
                .as_str()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();

            if node_type == "interface_type" || node_type == "interface" {
                let methods: HashMap<String, String> = struct_meta
                    .iter()
                    .filter(|(k, _)| k.starts_with("method:"))
                    .map(|(k, v)| (k.trim_start_matches("method:").to_string(), v.clone()))
                    .collect();
                interfaces.insert(name.clone(), methods);
            }

            nodes.insert(
                r.id.clone(),
                EntityNode { name, node_type, path, docstring, metadata: struct_meta },
            );
        }

        // Second pass — detect implementations.
        let mut impls: HashMap<String, Vec<String>> = HashMap::new();
        for (id, node) in &nodes {
            if node.node_type == "struct_type"
                || node.node_type == "struct"
                || node.node_type == "class"
            {
                for (iface_name, iface_methods) in &interfaces {
                    if iface_methods.is_empty() {
                        continue;
                    }
                    let implements = iface_methods
                        .keys()
                        .all(|m| node.metadata.contains_key(&format!("method:{m}")));
                    if implements {
                        impls.entry(iface_name.clone()).or_default().push(id.clone());
                    }
                }
            }
        }

        *self.nodes.write().unwrap() = nodes;
        *self.impls.write().unwrap() = impls;
    }

    pub fn get_implementations(&self, interface_name: &str) -> Vec<EntityNode> {
        let nodes = self.nodes.read().unwrap();
        let impls = self.impls.read().unwrap();
        impls
            .get(interface_name)
            .map(|ids| ids.iter().filter_map(|id| nodes.get(id).cloned()).collect())
            .unwrap_or_default()
    }

    pub fn find_usage(&self, field_name: &str) -> Vec<EntityNode> {
        let key = format!("field:{field_name}");
        self.nodes
            .read()
            .unwrap()
            .values()
            .filter(|n| n.metadata.contains_key(&key))
            .cloned()
            .collect()
    }

    pub fn search_by_name(&self, name: &str) -> Vec<EntityNode> {
        let lower = name.to_lowercase();
        self.nodes
            .read()
            .unwrap()
            .values()
            .filter(|n| n.name.to_lowercase().contains(&lower))
            .cloned()
            .collect()
    }
}
