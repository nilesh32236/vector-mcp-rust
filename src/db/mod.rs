use anyhow::{Context, Result};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::index::scalar::{FullTextSearchQuery, InvertedIndexParams as FtsIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table as LanceTable, connect};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Database Record Schema
// ---------------------------------------------------------------------------

/// The core data structure stored in LanceDB.
/// Used for both writing (indexing) and reading (searching).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub metadata: String, // JSON blob containing path, symbols, calls, etc.
}

impl Record {
    /// Returns the parsed metadata JSON object.
    pub fn metadata_json(&self) -> serde_json::Value {
        serde_json::from_str(&self.metadata).unwrap_or(serde_json::Value::Null)
    }

    /// Helper to get a specific metadata string field.
    pub fn metadata_str(&self, key: &str) -> String {
        self.metadata_json()[key].as_str().unwrap_or("").to_string()
    }

    /// Helper to get the content hash from metadata.
    pub fn content_hash(&self) -> String {
        self.metadata_str("content_hash")
    }
}

/// `Store` holds the LanceDB table handles.
pub struct Store {
    pub code_vectors: LanceTable,
    pub project_context: LanceTable,
}

impl Store {
    /// Open (or create) the core application table and return a ready `Store`.
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
        )
        .await?;

        Ok(Self {
            code_vectors,
            project_context,
        })
    }

    /// Perform a hybrid search (Vector + Full-Text Search).
    pub async fn hybrid_search(
        &self,
        vector: Vec<f32>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Record>> {
        let results = self
            .code_vectors
            .query()
            .nearest_to(vector)?
            .full_text_search(FullTextSearchQuery::new(query.to_string()))
            .limit(limit)
            .execute()
            .await?;

        let batches = results.try_collect::<Vec<_>>().await?;
        self.batches_to_records(batches).await
    }

    /// Delete all records associated with a specific file path.
    pub async fn delete_by_path(&self, path: &str) -> Result<()> {
        let predicate = format!(
            "metadata LIKE '%\"path\":\"{}\"%'",
            path.replace('\'', "''")
        );
        self.code_vectors
            .delete(&predicate)
            .await
            .context("Deleting records by path")
            .map(|_| ())
    }

    /// Retrieve records associated with a specific file path.
    pub async fn get_records_by_path(&self, path: &str) -> Result<Vec<Record>> {
        let predicate = format!(
            "metadata LIKE '%\"path\":\"{}\"%'",
            path.replace('\'', "''")
        );
        let results = self.code_vectors
            .query()
            .only_if(predicate)
            .execute()
            .await?;

        let batches = results.try_collect::<Vec<_>>().await?;
        self.batches_to_records(batches).await
    }

    /// Retrieve all records from the database (used for full codebase analysis).
    pub async fn get_all_records(&self) -> Result<Vec<Record>> {
        let results = self.code_vectors.query().execute().await?;

        let batches = results.try_collect::<Vec<_>>().await?;
        self.batches_to_records(batches).await
    }

    /// Helper: Converts Arrow RecordBatches back into our `Record` structs.

    pub async fn store_project_context(&self, project_id: &str, text: &str) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("project_id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
        ]));

        let id = uuid::Uuid::new_v4().to_string();
        let ids = arrow_array::StringArray::from(vec![id.as_str()]);
        let project_ids = arrow_array::StringArray::from(vec![project_id]);
        let texts = arrow_array::StringArray::from(vec![text]);

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(project_ids), Arc::new(texts)],
        )?;

        self.project_context.add(vec![batch]).execute().await?;
        Ok(())
    }

    pub async fn get_project_context(&self, project_id: &str) -> Result<Vec<String>> {
        let predicate = format!("project_id = '{}'", project_id.replace("'", "''"));
        let stream = self
            .project_context
            .query()
            .only_if(predicate)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect::<Vec<_>>().await?;

        let mut contexts = Vec::new();
        for batch in batches {
            let texts = batch
                .column(2)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .context("text column is not StringArray")?;

            for i in 0..batch.num_rows() {
                contexts.push(texts.value(i).to_string());
            }
        }
        Ok(contexts)
    }

    async fn batches_to_records(&self, batches: Vec<RecordBatch>) -> Result<Vec<Record>> {
        let mut records = Vec::new();
        for batch in batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .context("id column is not StringArray")?;
            let contents = batch
                .column(1)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .context("content column is not StringArray")?;
            let vectors = batch
                .column(2)
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeListArray>()
                .context("vector column is not FixedSizeListArray")?;
            let metadatas = batch
                .column(3)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .context("metadata column is not StringArray")?;

            for i in 0..batch.num_rows() {
                let vec_val = vectors.value(i);
                let vec_f32 = vec_val
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .context("vector value is not Float32Array")?
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
}

pub async fn connect_store(uri: &str, dimension: usize) -> Result<Store> {
    let connection = connect(uri)
        .execute()
        .await
        .with_context(|| format!("connecting to LanceDB at {uri}"))?;

    Store::open_tables(connection, dimension).await
}

async fn open_or_create_table(
    connection: &Connection,
    name: &str,
    empty_batch: RecordBatch,
) -> Result<LanceTable> {
    let table_names = connection.table_names().execute().await?;
    if table_names.contains(&name.to_string()) {
        connection
            .open_table(name)
            .execute()
            .await
            .context("Opening table")
    } else {
        let table = connection
            .create_table(name, empty_batch)
            .execute()
            .await
            .context("Creating table")?;

        // Create Full-Text Search index on the content column for hybrid search.
        table
            .create_index(&["content"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await
            .context("Creating FTS index")?;

        Ok(table)
    }
}
