use std::sync::Arc;
use anyhow::{Context, Result};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::index::scalar::{FullTextSearchQuery, InvertedIndexParams as FtsIndexBuilder};
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, Table as LanceTable};
use serde::{Deserialize, Serialize};

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

/// `Store` holds the LanceDB table handles.
pub struct Store {
    pub code_vectors: LanceTable,
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

        Ok(Self { code_vectors })
    }

    /// Perform a hybrid search (Vector + Full-Text Search).
    pub async fn hybrid_search(&self, vector: Vec<f32>, query: &str, limit: usize) -> Result<Vec<Record>> {
        let results = self.code_vectors
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
        let predicate = format!("metadata LIKE '%\"path\":\"{}\"%'", path.replace('\'', "''"));
        self.code_vectors.delete(&predicate).await.context("Deleting records by path").map(|_| ())
    }

    /// Helper: Converts Arrow RecordBatches back into our `Record` structs.
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
        connection.open_table(name).execute().await.context("Opening table")
    } else {
        let table = connection
            .create_table(name, empty_batch)
            .execute()
            .await
            .context("Creating table")?;
        
        // Create Full-Text Search index on the content column for hybrid search.
        table.create_index(&["content"], Index::FTS(FtsIndexBuilder::default())).execute().await.context("Creating FTS index")?;
        
        Ok(table)
    }
}
