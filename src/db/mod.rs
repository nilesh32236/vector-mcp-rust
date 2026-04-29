#![allow(dead_code)]
pub mod graph;
use crate::llm::embedding::Embedder;
use anyhow::{Context, Result, bail};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table as LanceTable, connect};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::{Arc, RwLock};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field as TantivyField, STORED, STRING, Schema as TantivySchema, TEXT, Term,
    Value as TantivyValue,
};
use tantivy::{Index as TantivyIndex, IndexReader, IndexWriter, ReloadPolicy, doc};
use tracing::info;

// ---------------------------------------------------------------------------
// Tantivy Lexical Index
// ---------------------------------------------------------------------------

pub struct LexicalIndex {
    index: TantivyIndex,
    reader: IndexReader,
    writer: Option<Arc<RwLock<IndexWriter>>>,
    fields: LexicalFields,
}

struct LexicalFields {
    id: TantivyField,
    text: TantivyField,
}

impl LexicalIndex {
    fn open_or_create(path: &Path, read_only: bool) -> Result<Self> {
        let mut schema_builder = TantivySchema::builder();
        let id = schema_builder.add_text_field("id", STRING | STORED);
        let text = schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();

        let index = if path.exists() && path.is_dir() && std::fs::read_dir(path)?.next().is_some() {
            TantivyIndex::open_in_dir(path).context("Opening Tantivy index")?
        } else {
            if read_only {
                bail!("Cannot create a new index in read-only mode");
            }
            std::fs::create_dir_all(path).context("Creating index directory")?;
            TantivyIndex::create_in_dir(path, schema).context("Creating Tantivy index")?
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        let writer = if read_only {
            None
        } else {
            let w = match index.writer(50_000_000) {
                Ok(w) => w,
                Err(tantivy::TantivyError::LockFailure(_, _)) => {
                    // Stale lock from a previous crashed process — delete and retry.
                    tracing::warn!("Tantivy lock busy, removing stale lockfile and retrying");
                    let lock_path = path.join(".tantivy-writer.lock");
                    let _ = std::fs::remove_file(&lock_path);
                    index
                        .writer(50_000_000)
                        .context("Acquiring Tantivy writer after lock removal")?
                }
                Err(e) => return Err(e).context("Acquiring Tantivy writer"),
            };
            Some(Arc::new(RwLock::new(w)))
        };

        Ok(Self {
            index,
            reader,
            writer,
            fields: LexicalFields { id, text },
        })
    }

    fn add(&self, doc_id: &str, text: &str) -> Result<()> {
        let Some(w) = &self.writer else {
            return Ok(());
        };
        let mut writer = w.write().unwrap();
        writer.delete_term(Term::from_field_text(self.fields.id, doc_id));
        writer.add_document(doc!(
            self.fields.id => doc_id,
            self.fields.text => text,
        ))?;
        writer.commit()?;
        Ok(())
    }

    /// Add without committing — call `commit()` after a batch.
    fn add_no_commit(&self, doc_id: &str, text: &str) -> Result<()> {
        let Some(w) = &self.writer else {
            return Ok(());
        };
        let writer = w.write().unwrap();
        writer.delete_term(Term::from_field_text(self.fields.id, doc_id));
        writer.add_document(doc!(
            self.fields.id => doc_id,
            self.fields.text => text,
        ))?;
        Ok(())
    }

    fn commit(&self) -> Result<()> {
        if let Some(w) = &self.writer {
            w.write().unwrap().commit()?;
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.reader.searcher().num_docs() == 0
    }

    fn remove(&self, doc_id: &str) -> Result<()> {
        let Some(w) = &self.writer else {
            return Ok(());
        };
        let mut writer = w.write().unwrap();
        writer.delete_term(Term::from_field_text(self.fields.id, doc_id));
        writer.commit()?;
        Ok(())
    }

    fn remove_no_commit(&self, doc_id: &str) -> Result<()> {
        let Some(w) = &self.writer else {
            return Ok(());
        };
        let writer = w.write().unwrap();
        writer.delete_term(Term::from_field_text(self.fields.id, doc_id));
        Ok(())
    }

    fn search(&self, query_str: &str, top_k: usize) -> Vec<(String, f32)> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.text]);

        let query = match query_parser.parse_query(query_str) {
            Ok(q) => q,
            Err(_) => return vec![],
        };

        let top_docs = match searcher.search(&query, &TopDocs::with_limit(top_k)) {
            Ok(docs) => docs,
            Err(_) => return vec![],
        };

        top_docs
            .into_iter()
            .filter_map(|(score, doc_address)| {
                let retrieved_doc = searcher.doc::<tantivy::TantivyDocument>(doc_address).ok()?;
                let id = retrieved_doc
                    .get_first(self.fields.id)?
                    .as_str()?
                    .to_string();
                Some((id, score))
            })
            .collect()
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
    pub quantized_code: Option<Vec<u8>>,
}

impl Record {
    pub fn metadata_json(&self) -> serde_json::Value {
        serde_json::from_str(&self.metadata).unwrap_or(serde_json::Value::Null)
    }

    pub fn metadata_str(&self, key: &str) -> String {
        self.metadata_json()[key].as_str().unwrap_or("").to_string()
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
    lexical: Arc<LexicalIndex>,
    pub graph: graph::KnowledgeGraph,
    pub quantized_cache: Arc<dashmap::DashMap<String, Vec<u8>>>,
}

impl Store {
    async fn open_tables(
        connection: Connection,
        dimension: usize,
        index_path: &Path,
    ) -> Result<Self> {
        let code_vectors = open_or_create_table(
            &connection,
            "code_vectors",
            RecordBatch::new_empty(Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Utf8, false),
                ArrowField::new("content", DataType::Utf8, false),
                ArrowField::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        dimension as i32,
                    ),
                    false,
                ),
                ArrowField::new("metadata", DataType::Utf8, true),
                ArrowField::new("quantized_code", DataType::Binary, true),
            ]))),
            dimension,
        )
        .await?;

        let project_context = open_or_create_table(
            &connection,
            "project_context",
            RecordBatch::new_empty(Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Utf8, false),
                ArrowField::new("project_id", DataType::Utf8, false),
                ArrowField::new("text", DataType::Utf8, false),
            ]))),
            0, // no vector index needed
        )
        .await?;

        let lexical = Arc::new(LexicalIndex::open_or_create(index_path, false)?);

        let store = Self {
            code_vectors,
            project_context,
            lexical,
            graph: graph::KnowledgeGraph::new(),
            quantized_cache: Arc::new(dashmap::DashMap::new()),
        };

        if store.code_vectors.count_rows(None).await? > 0 {
            info!(
                "Existing records found. Synchronizing in-memory graph, lexical index, and quantized cache..."
            );
            if let Ok(records) = store.get_all_records().await {
                store.graph.populate(&records);
                for r in &records {
                    if let Some(code) = &r.quantized_code {
                        store.quantized_cache.insert(r.id.clone(), code.clone());
                    }
                }
                // Rebuild Tantivy if it has no segments (e.g. fresh install or schema change).
                if store.lexical.is_empty() {
                    info!(
                        "Tantivy index is empty — rebuilding from {} existing records.",
                        records.len()
                    );
                    for r in &records {
                        let _ = store.lexical.add_no_commit(&r.id, &bm25_document_text(r));
                    }
                    let _ = store.lexical.commit();
                }
            }
        }

        Ok(store)
    }

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    pub async fn get_record_by_id(&self, id: &str) -> Result<Option<Record>> {
        let predicate = format!("id = '{}'", sql_escape(id));
        let results = self
            .code_vectors
            .query()
            .only_if(predicate)
            .execute()
            .await?;
        let batches = results.try_collect::<Vec<_>>().await?;
        let mut records = batches_to_records(batches)?;
        Ok(records.pop())
    }

    pub async fn get_records_by_ids(&self, ids: &[String]) -> Result<HashMap<String, Record>> {
        if ids.is_empty() || self.code_vectors.count_rows(None).await? == 0 {
            return Ok(HashMap::new());
        }

        let predicate = ids
            .iter()
            .map(|id| format!("id = '{}'", sql_escape(id)))
            .collect::<Vec<_>>()
            .join(" OR ");

        let results = self
            .code_vectors
            .query()
            .only_if(predicate)
            .execute()
            .await?;
        let batches = results.try_collect::<Vec<_>>().await?;
        Ok(batches_to_records(batches)?
            .into_iter()
            .map(|r| (r.id.clone(), r))
            .collect())
    }

    pub async fn update_record_metadata(&self, record_id: &str, new_metadata: &str) -> Result<()> {
        let escaped_id = sql_escape(record_id);
        let escaped_meta = sql_escape(new_metadata);
        self.code_vectors
            .update()
            .only_if(format!("id = '{escaped_id}'"))
            .column("metadata", format!("'{escaped_meta}'"))
            .execute()
            .await
            .context("Updating record metadata")?;
        Ok(())
    }

    pub async fn upsert_records(&self, records: Vec<Record>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let batch = records_to_batch(&records)?;

        for r in &records {
            let _ = self.lexical.add_no_commit(&r.id, &bm25_document_text(r));
            self.graph.add_record(r);
            if let Some(code) = &r.quantized_code {
                self.quantized_cache.insert(r.id.clone(), code.clone());
            }
        }
        let _ = self.lexical.commit();

        self.code_vectors.add(vec![batch]).execute().await?;
        Ok(())
    }

    pub async fn insert_batch(&self, records: &[Record]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let batch = records_to_batch(records)?;
        for r in records {
            let _ = self.lexical.add_no_commit(&r.id, &bm25_document_text(r));
            self.graph.add_record(r);
            if let Some(code) = &r.quantized_code {
                self.quantized_cache.insert(r.id.clone(), code.clone());
            }
        }
        let _ = self.lexical.commit();
        self.code_vectors.add(vec![batch]).execute().await?;
        Ok(())
    }

    pub fn commit_lexical(&self) -> Result<()> {
        self.lexical.commit()
    }

    pub async fn delete_by_path(&self, path: &str) -> Result<()> {
        self.delete_by_path_inner(path, true).await
    }

    pub async fn delete_by_path_no_commit(&self, path: &str) -> Result<()> {
        self.delete_by_path_inner(path, false).await
    }

    async fn delete_by_path_inner(&self, path: &str, commit: bool) -> Result<()> {
        let predicate = format!("metadata LIKE '%\"path\":\"{}\"%'", sql_escape(path));

        let to_delete = self.get_records_by_path(path).await?;
        if to_delete.is_empty() {
            return Ok(());
        }

        self.code_vectors
            .delete(&predicate)
            .await
            .context("Deleting records by path")?;

        for r in to_delete {
            self.quantized_cache.remove(&r.id);
            if commit {
                let _ = self.lexical.remove(&r.id);
            } else {
                let _ = self.lexical.remove_no_commit(&r.id);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    pub async fn vector_search(
        &self,
        vector: Vec<f32>,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<Record>> {
        let mut query = self.code_vectors.query().nearest_to(vector)?.limit(limit);

        if let Some(pids) = project_ids
            && !pids.is_empty()
        {
            let parts: Vec<String> = pids
                .iter()
                .map(|p| format!("metadata LIKE '%\"project_id\":\"{}\"%'", sql_escape(p)))
                .collect();
            let predicate = parts.join(" OR ");
            query = query.only_if(predicate);
        }

        let results = query.execute().await?;
        let batches = results.try_collect::<Vec<_>>().await?;
        batches_to_records(batches)
    }

    pub async fn lexical_search(
        &self,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<Record>> {
        let hits = self.lexical.search(query, limit * 3);
        if hits.is_empty() {
            return Ok(vec![]);
        }

        let ids = hits.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
        let records = self.get_records_by_ids(&ids).await?;
        let mut matched: Vec<(Record, f32)> = Vec::new();
        for (id, score) in hits {
            if let Some(r) = records.get(&id) {
                let r = r.clone();
                if let Some(pids) = project_ids {
                    if !pids.is_empty() {
                        let pid = r.metadata_str("project_id");
                        if !pids.contains(&pid) {
                            continue;
                        }
                    }
                }
                matched.push((r, score));
            }
        }

        matched.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matched.truncate(limit);
        Ok(matched.into_iter().map(|(r, _)| r).collect())
    }

    pub async fn hybrid_search(
        &self,
        vector: Vec<f32>,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<Record>> {
        Ok(self
            .hybrid_search_scored(vector, query, limit, project_ids)
            .await?
            .into_iter()
            .map(|(record, _)| record)
            .collect())
    }

    pub async fn hybrid_search_scored(
        &self,
        vector: Vec<f32>,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<(Record, f32)>> {
        let fetch = limit * 3;

        let (vec_res, lex_res) = tokio::join!(
            self.vector_search(vector, fetch, project_ids),
            self.lexical_search(query, fetch, project_ids),
        );

        let vec_results = vec_res.unwrap_or_default();
        let lex_results = lex_res.unwrap_or_default();

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
            .filter_map(|(id, score)| record_map.remove(&id).map(|r| (r, score as f32)))
            .collect())
    }

    pub async fn turbo_search(
        &self,
        embedder: &Embedder,
        query_vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<(Record, f32)>> {
        #[derive(PartialEq)]
        struct SearchHit {
            id: String,
            score: f32,
        }
        impl Eq for SearchHit {}
        impl Ord for SearchHit {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for SearchHit {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::with_capacity(limit + 1);

        for entry in self.quantized_cache.iter() {
            let (id, code) = entry.pair();
            if let Ok(score) = embedder.estimate_similarity(code, &query_vector) {
                heap.push(SearchHit {
                    id: id.clone(),
                    score,
                });
                if heap.len() > limit {
                    heap.pop();
                }
            }
        }

        let mut ordered_hits = Vec::new();
        while let Some(hit) = heap.pop() {
            ordered_hits.push(hit);
        }
        let ids = ordered_hits
            .iter()
            .map(|hit| hit.id.clone())
            .collect::<Vec<_>>();
        let records = self.get_records_by_ids(&ids).await?;
        let mut final_hits = ordered_hits
            .into_iter()
            .filter_map(|hit| records.get(&hit.id).cloned().map(|r| (r, hit.score)))
            .collect::<Vec<_>>();
        final_hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(final_hits)
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
        let predicate = format!("metadata LIKE '%\"path\":\"{}\"%'", sql_escape(path));
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
    // Project context
    // -----------------------------------------------------------------------

    pub async fn store_project_context(&self, project_id: &str, text: &str) -> Result<()> {
        use arrow_array::StringArray;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Utf8, false),
            ArrowField::new("project_id", DataType::Utf8, false),
            ArrowField::new("text", DataType::Utf8, false),
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
        let predicate = format!("project_id = '{}'", sql_escape(project_id));
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

pub async fn connect_store(uri: &str, dimension: usize, read_only: bool) -> Result<Store> {
    let final_uri = if uri.starts_with('/') {
        format!("file://{}", uri)
    } else {
        uri.to_string()
    };

    let path_str = final_uri.trim_start_matches("file://");
    let tantivy_path = Path::new(path_str).join("lexical_index");

    let connection = connect(&final_uri)
        .execute()
        .await
        .with_context(|| format!("connecting to LanceDB at {final_uri}"))?;

    let mut store = Store::open_tables(connection, dimension, &tantivy_path).await?;
    if read_only {
        // Re-open lexical index in read-only mode
        store.lexical = Arc::new(LexicalIndex::open_or_create(&tantivy_path, true)?);
    }
    Ok(store)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Escape a string for use inside a single-quoted SQL literal.
/// Escapes both single quotes (SQL standard) and backslashes (some dialects).
fn sql_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "''")
}

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
    _dimension: usize,
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

    // TODO: move index creation to background task (only when count_rows > 5000)
    /*
    if dimension > 0 {
        if let Err(e) = table
            .create_index(
                &["vector"],
                LanceIndex::IvfPq(
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
    */

    Ok(table)
}

fn records_to_batch(records: &[Record]) -> Result<RecordBatch> {
    use arrow_array::{FixedSizeListArray, Float32Array, StringArray};
    use lance_arrow::FixedSizeListArrayExt;

    let dimension = records.first().map(|r| r.vector.len()).unwrap_or(0);

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Utf8, false),
        ArrowField::new("content", DataType::Utf8, false),
        ArrowField::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                dimension as i32,
            ),
            false,
        ),
        ArrowField::new("metadata", DataType::Utf8, true),
        ArrowField::new("quantized_code", DataType::Binary, true),
    ]));

    let ids = StringArray::from(records.iter().map(|r| r.id.as_str()).collect::<Vec<_>>());
    let contents = StringArray::from(
        records
            .iter()
            .map(|r| r.content.as_str())
            .collect::<Vec<_>>(),
    );
    let metadatas = StringArray::from(
        records
            .iter()
            .map(|r| r.metadata.as_str())
            .collect::<Vec<_>>(),
    );
    let quantized_codes = arrow_array::BinaryArray::from(
        records
            .iter()
            .map(|r| r.quantized_code.as_deref())
            .collect::<Vec<_>>(),
    );

    let flat: Vec<f32> = records
        .iter()
        .flat_map(|r| r.vector.iter().copied())
        .collect();
    let vector_values = Float32Array::from(flat);
    let vectors = FixedSizeListArray::try_new_from_values(vector_values, dimension as i32)?;

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(contents),
            Arc::new(vectors),
            Arc::new(metadatas),
            Arc::new(quantized_codes),
        ],
    )
    .context("Building RecordBatch")
}

pub fn batches_to_records(batches: Vec<RecordBatch>) -> Result<Vec<Record>> {
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

        let quantized_codes = if batch.num_columns() > 4 {
            batch
                .column(4)
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
        } else {
            None
        };

        for i in 0..batch.num_rows() {
            let vec_val = vectors.value(i);
            let vec_f32 = vec_val
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .context("vector value")?
                .values()
                .to_vec();

            let q_code = quantized_codes.and_then(|qc| {
                if qc.is_null(i) {
                    None
                } else {
                    Some(qc.value(i).to_vec())
                }
            });

            records.push(Record {
                id: ids.value(i).to_string(),
                content: contents.value(i).to_string(),
                vector: vec_f32,
                metadata: metadatas.value(i).to_string(),
                quantized_code: q_code,
            });
        }
    }
    Ok(records)
}
