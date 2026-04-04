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
    /// interface_name -> method_name -> signature/meta
    interfaces: RwLock<HashMap<String, HashMap<String, String>>>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            impls: RwLock::new(HashMap::new()),
            interfaces: RwLock::new(HashMap::new()),
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

        {
            let mut nodes = self.nodes.write().unwrap();
            nodes.insert(r.id.clone(), node.clone());
        }

        if node_type == "interface_type" || node_type == "interface" {
            let methods: HashMap<String, String> = struct_meta
                .iter()
                .filter(|(k, _)| k.starts_with("method:"))
                .map(|(k, v)| (k.trim_start_matches("method:").to_string(), v.clone()))
                .collect();

            self.interfaces
                .write()
                .unwrap()
                .insert(name.clone(), methods.clone());

            // Re-check all structs against this new interface
            let mut impls = self.impls.write().unwrap();
            let nodes = self.nodes.read().unwrap();
            let mut new_impls = Vec::new();
            for (id, other_node) in nodes.iter() {
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
            impls.insert(name, new_impls);
        } else if node_type == "struct_type" || node_type == "struct" || node_type == "class" {
            // Check this new struct against all known interfaces
            let interfaces = self.interfaces.read().unwrap();
            let mut impls = self.impls.write().unwrap();
            for (iface_name, iface_methods) in interfaces.iter() {
                if iface_methods.is_empty() {
                    continue;
                }
                let implements = iface_methods
                    .keys()
                    .all(|m| node.metadata.contains_key(&format!("method:{m}")));
                let iface_impls = impls.entry(iface_name.clone()).or_default();
                if implements {
                    if !iface_impls.contains(&r.id) {
                        iface_impls.push(r.id.clone());
                    }
                } else {
                    iface_impls.retain(|id| id != &r.id);
                }
            }
        }
    }

    /// Rebuild the graph from a fresh set of DB records.
    pub fn populate(&self, records: &[Record]) {
        // Clear existing state
        *self.nodes.write().unwrap() = HashMap::new();
        *self.impls.write().unwrap() = HashMap::new();
        *self.interfaces.write().unwrap() = HashMap::new();

        for r in records {
            self.add_record(r);
        }
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
