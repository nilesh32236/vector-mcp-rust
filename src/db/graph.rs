use crate::db::Record;
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
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            nodes: DashMap::new(),
            impls: DashMap::new(),
            interfaces: DashMap::new(),
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

        if node_type == "interface_type" || node_type == "interface" {
            let methods: HashMap<String, String> = struct_meta
                .iter()
                .filter(|(k, _)| k.starts_with("method:"))
                .map(|(k, v)| (k.trim_start_matches("method:").to_string(), v.clone()))
                .collect();

            self.interfaces.insert(name.clone(), methods.clone());

            // Re-check all structs against this new interface
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
            // Check this new struct against all known interfaces
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
    }

    /// Rebuild the graph from a fresh set of DB records.
    pub fn populate(&self, records: &[Record]) {
        self.nodes.clear();
        self.impls.clear();
        self.interfaces.clear();

        for r in records {
            self.add_record(r);
        }
    }

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
}
