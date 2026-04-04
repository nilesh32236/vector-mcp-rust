#![allow(dead_code)]
pub mod graph;
use anyhow::{Context, Result};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::index::scalar::{InvertedIndexParams as FtsIndexBuilder};
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table as LanceTable, connect};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::warn;

// ---------------------------------------------------------------------------
// BM25 Lexical Index
// ---------------------------------------------------------------------------

/// Minimal in-memory BM25 index (k1=1.5, b=0.75) matching Go's lexical.Index.
struct Bm25Index {
    /// term → {doc_id → term_freq}
    inverted: HashMap<String, HashMap<String, f32>>,
    /// doc_id → doc_length (token count)
    doc_lengths: HashMap<String, usize>,
    avg_doc_len: f32,
    k1: f32,
    b: f32,
}

impl Bm25Index {
    fn new() -> Self {
        Self {
            inverted: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_doc_len: 0.0,
            k1: 1.5,
            b: 0.75,
        }
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 1)
            .map(String::from)
            .collect()
    }

    fn add(&mut self, doc_id: &str, text: &str) {
        let tokens = Self::tokenize(text);
        let len = tokens.len();
        self.doc_lengths.insert(doc_id.to_string(), len);

        let mut freq: HashMap<String, f32> = HashMap::new();
        for t in tokens {
            *freq.entry(t).or_default() += 1.0;
        }
        for (term, tf) in freq {
            self.inverted
                .entry(term)
                .or_default()
                .insert(doc_id.to_string(), tf);
        }
        self.recompute_avg();
    }

    fn remove(&mut self, doc_id: &str) {
        self.doc_lengths.remove(doc_id);
        for postings in self.inverted.values_mut() {
            postings.remove(doc_id);
        }
        self.recompute_avg();
    }

    fn recompute_avg(&mut self) {
        if self.doc_lengths.is_empty() {
            self.avg_doc_len = 0.0;
        } else {
            let total: usize = self.doc_lengths.values().sum();
            self.avg_doc_len = total as f32 / self.doc_lengths.len() as f32;
        }
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let terms = Self::tokenize(query);
        let n = self.doc_lengths.len() as f32;
        if n == 0.0 || terms.is_empty() {
            return vec![];
        }

        let mut scores: HashMap<String, f32> = HashMap::new();
        for term in &terms {
            let Some(postings) = self.inverted.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for (doc_id, &tf) in postings {
                let dl = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f32;
                let norm = 1.0 - self.b + self.b * dl / self.avg_doc_len.max(1.0);
                let score = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm);
                *scores.entry(doc_id.clone()).or_default() += score;
            }
        }

        let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }
}

// ---------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub content: String,
    pub vector: Vec<f32>,
    /// JSON blob — matches Go's `map[string]string` serialised with `json.Marshal`.
    pub metadata: String,
}

impl Record {
    pub fn metadata_json(&self) -> serde_json::Value {
        serde_json::from_str(&self.metadata).unwrap_or(serde_json::Value::Null)
    }

    pub fn metadata_str(&self, key: &str) -> String {
        self.metadata_json()[key]
            .as_str()
            .unwrap_or("")
            .to_string()
    }

    pub fn content_hash(&self) -> String {
        self.metadata_str("content_hash")
    }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

pub struct Store {
    pub code_vectors: LanceTable,
    pub project_context: LanceTable,
    bm25: RwLock<Bm25Index>,
    pub graph: graph::KnowledgeGraph,
}

impl Store {
    async fn open_tables(connection: Connection, dimension: usize) -> Result<Self> {
        let code_vectors = open_or_create_table(
            &connection,
            "code_vectors",
            RecordBatch::new_empty(Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("content", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimension as i32,
                    ),
                    false,
                ),
                Field::new("metadata", DataType::Utf8, true),
            ]))),
            dimension,
        )
        .await?;

        let project_context = open_or_create_table(
            &connection,
            "project_context",
            RecordBatch::new_empty(Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("project_id", DataType::Utf8, false),
                Field::new("text", DataType::Utf8, false),
            ]))),
            0, // no vector index needed
        )
        .await?;

        let store = Self {
            code_vectors,
            project_context,
            bm25: RwLock::new(Bm25Index::new()),
            graph: graph::KnowledgeGraph::new(),
        };

        // Bootstrap BM25 and knowledge graph from existing records.
        if let Ok(records) = store.get_all_records().await {
            {
                let mut idx = store.bm25.write().unwrap();
                for r in &records {
                    idx.add(&r.id, &bm25_document_text(r));
                }
            } // guard dropped before next await
            store.graph.populate(&records);
        }

        Ok(store)
    }

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /// Upsert a batch of records: delete existing by path then insert.
    pub async fn upsert_records(&self, records: Vec<Record>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let batch = records_to_batch(&records)?;

        // Update BM25 before the await so the guard is dropped.
        {
            let mut idx = self.bm25.write().unwrap();
            for r in &records {
                idx.add(&r.id, &bm25_document_text(r));
            }
        } // guard dropped here

        self.code_vectors.add(vec![batch]).execute().await?;

        // Refresh graph after the await.
        if let Ok(all) = self.get_all_records().await {
            self.graph.populate(&all);
        }
        Ok(())
    }

    pub async fn delete_by_path(&self, path: &str) -> Result<()> {
        let predicate = format!(
            "metadata LIKE '%\"path\":\"{}\"%'",
            path.replace('\'', "''")
        );
        self.code_vectors
            .delete(&predicate)
            .await
            .context("Deleting records by path")
            .map(|_| ())?;

        // Remove from BM25 — we don't have IDs here so rebuild lazily on next search.
        // For correctness, remove any doc whose path matches.
        let mut idx = self.bm25.write().unwrap();
        let to_remove: Vec<String> = idx
            .doc_lengths
            .keys()
            .filter(|id| id.contains(path))
            .cloned()
            .collect();
        for id in to_remove {
            idx.remove(&id);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Vector-only ANN search.
    pub async fn vector_search(
        &self,
        vector: Vec<f32>,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<Record>> {
        let mut query = self.code_vectors.query().nearest_to(vector)?.limit(limit);

        if let Some(pids) = project_ids
            && !pids.is_empty() {
                let parts: Vec<String> = pids
                    .iter()
                    .map(|p| format!("metadata LIKE '%\"project_id\":\"{}\"%'", p.replace('\'', "''")))
                    .collect();
                let predicate = parts.join(" OR ");
                query = query.only_if(predicate);
            }

        let results = query.execute().await?;
        let batches = results.try_collect::<Vec<_>>().await?;
        batches_to_records(batches)
    }

    /// Lexical BM25 search backed by in-memory index.
    pub async fn lexical_search(
        &self,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<Record>> {
        let hits = {
            let idx = self.bm25.read().unwrap();
            idx.search(query, limit * 3)
        };
        if hits.is_empty() {
            return Ok(vec![]);
        }

        let all = self.get_all_records().await?;
        let id_set: HashMap<&str, f32> = hits.iter().map(|(id, s)| (id.as_str(), *s)).collect();

        let mut matched: Vec<(Record, f32)> = all
            .into_iter()
            .filter(|r| {
                if let Some(pids) = project_ids {
                    if pids.is_empty() { return true; }
                    let pid = r.metadata_str("project_id");
                    pids.contains(&pid)
                } else {
                    true
                }
            })
            .filter_map(|r| {
                let id = r.id.clone();
                id_set.get(id.as_str()).map(|&s| (r, s))
            })
            .collect();

        matched.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matched.truncate(limit);
        Ok(matched.into_iter().map(|(r, _)| r).collect())
    }

    /// Hybrid search: RRF fusion of vector ANN + BM25 lexical, matching Go's implementation.
    pub async fn hybrid_search(
        &self,
        vector: Vec<f32>,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<Record>> {
        let fetch = limit * 3;

        // Run both searches concurrently.
        let (vec_res, lex_res) = tokio::join!(
            self.vector_search(vector, fetch, project_ids),
            self.lexical_search(query, fetch, project_ids),
        );

        let vec_results = vec_res.unwrap_or_default();
        let lex_results = lex_res.unwrap_or_default();

        // Reciprocal Rank Fusion (k=60) — same as Go.
        let k = 60.0_f64;
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut record_map: HashMap<String, Record> = HashMap::new();

        for (i, r) in vec_results.into_iter().enumerate() {
            *scores.entry(r.id.clone()).or_default() += 1.0 / (k + (i + 1) as f64);
            record_map.insert(r.id.clone(), r);
        }
        for (i, r) in lex_results.into_iter().enumerate() {
            *scores.entry(r.id.clone()).or_default() += 1.0 / (k + (i + 1) as f64);
            record_map.entry(r.id.clone()).or_insert(r);
        }

        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(limit);

        Ok(ranked
            .into_iter()
            .filter_map(|(id, _)| record_map.remove(&id))
            .collect())
    }

    // -----------------------------------------------------------------------
    // Bulk reads
    // -----------------------------------------------------------------------

    pub async fn get_all_records(&self) -> Result<Vec<Record>> {
        if self.code_vectors.count_rows(None).await? == 0 {
            return Ok(vec![]);
        }
        let results = self.code_vectors.query().execute().await?;
        let batches = results.try_collect::<Vec<_>>().await?;
        batches_to_records(batches)
    }

    pub async fn get_records_by_path(&self, path: &str) -> Result<Vec<Record>> {
        if self.code_vectors.count_rows(None).await? == 0 {
            return Ok(vec![]);
        }
        let predicate = format!(
            "metadata LIKE '%\"path\":\"{}\"%'",
            path.replace('\'', "''")
        );
        let results = self
            .code_vectors
            .query()
            .only_if(predicate)
            .execute()
            .await?;
        let batches = results.try_collect::<Vec<_>>().await?;
        batches_to_records(batches)
    }

    // -----------------------------------------------------------------------
    // Project context (knowledge items)
    // -----------------------------------------------------------------------

    pub async fn store_project_context(&self, project_id: &str, text: &str) -> Result<()> {
        use arrow_array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("project_id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let id = uuid::Uuid::new_v4().to_string();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![id.as_str()])),
                Arc::new(StringArray::from(vec![project_id])),
                Arc::new(StringArray::from(vec![text])),
            ],
        )?;
        self.project_context.add(vec![batch]).execute().await?;
        Ok(())
    }

    pub async fn get_project_context(&self, project_id: &str) -> Result<Vec<String>> {
        let predicate = format!("project_id = '{}'", project_id.replace('\'', "''"));
        let stream = self
            .project_context
            .query()
            .only_if(predicate)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut out = Vec::new();
        for batch in batches {
            let texts = batch
                .column(2)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .context("text column is not StringArray")?;
            for i in 0..batch.num_rows() {
                out.push(texts.value(i).to_string());
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Public constructor
// ---------------------------------------------------------------------------

pub async fn connect_store(uri: &str, dimension: usize) -> Result<Store> {
    let final_uri = if uri.starts_with('/') {
        format!("file://{}", uri)
    } else {
        uri.to_string()
    };
    let connection = connect(&final_uri)
        .execute()
        .await
        .with_context(|| format!("connecting to LanceDB at {final_uri}"))?;
    Store::open_tables(connection, dimension).await
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Text fed into BM25 — mirrors Go's `lexicalDocumentText`.
fn bm25_document_text(r: &Record) -> String {
    let meta = r.metadata_json();
    let mut parts = vec![r.content.clone()];
    for key in &["path", "symbols", "calls"] {
        if let Some(v) = meta[key].as_str() {
            parts.push(v.to_string());
        } else if let Some(arr) = meta[key].as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    parts.push(s.to_string());
                }
            }
        }
    }
    parts.join(" ")
}

async fn open_or_create_table(
    connection: &Connection,
    name: &str,
    empty_batch: RecordBatch,
    dimension: usize,
) -> Result<LanceTable> {
    let table_names = connection.table_names().execute().await?;
    if table_names.contains(&name.to_string()) {
        return connection
            .open_table(name)
            .execute()
            .await
            .context("Opening table");
    }

    let table = connection
        .create_table(name, empty_batch)
        .execute()
        .await
        .context("Creating table")?;

    if dimension > 0 {
        // Full-Text Search index on content for lexical hybrid search.
        if let Err(e) = table
            .create_index(&["content"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await
        {
            warn!("FTS index creation failed (non-fatal): {e}");
        }

        // IvfPq ANN vector index — matches Go's HNSW intent but uses the
        // lancedb Rust crate's available builder.
        if let Err(e) = table
            .create_index(
                &["vector"],
                Index::IvfPq(
                    IvfPqIndexBuilder::default()
                        .num_partitions(256)
                        .num_sub_vectors(16),
                ),
            )
            .execute()
            .await
        {
            warn!("IvfPq index creation failed (non-fatal, table may be empty): {e}");
        }
    }

    Ok(table)
}

fn records_to_batch(records: &[Record]) -> Result<RecordBatch> {
    use arrow_array::{FixedSizeListArray, Float32Array, StringArray};
    use lance_arrow::FixedSizeListArrayExt;

    // Infer dimension from first record.
    let dimension = records
        .first()
        .map(|r| r.vector.len())
        .unwrap_or(0);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension as i32,
            ),
            false,
        ),
        Field::new("metadata", DataType::Utf8, true),
    ]));

    let ids = StringArray::from(records.iter().map(|r| r.id.as_str()).collect::<Vec<_>>());
    let contents =
        StringArray::from(records.iter().map(|r| r.content.as_str()).collect::<Vec<_>>());
    let metadatas =
        StringArray::from(records.iter().map(|r| r.metadata.as_str()).collect::<Vec<_>>());

    let flat: Vec<f32> = records.iter().flat_map(|r| r.vector.iter().copied()).collect();
    let vector_values = Float32Array::from(flat);
    let vectors = FixedSizeListArray::try_new_from_values(vector_values, dimension as i32)?;

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(contents),
            Arc::new(vectors),
            Arc::new(metadatas),
        ],
    )
    .context("Building RecordBatch")
}

fn batches_to_records(batches: Vec<RecordBatch>) -> Result<Vec<Record>> {
    let mut records = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .context("id column")?;
        let contents = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .context("content column")?;
        let vectors = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .context("vector column")?;
        let metadatas = batch
            .column(3)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .context("metadata column")?;

        for i in 0..batch.num_rows() {
            let vec_val = vectors.value(i);
            let vec_f32 = vec_val
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .context("vector value")?
                .values()
                .to_vec();

            records.push(Record {
                id: ids.value(i).to_string(),
                content: contents.value(i).to_string(),
                vector: vec_f32,
                metadata: metadatas.value(i).to_string(),
            });
        }
    }
    Ok(records)
}
