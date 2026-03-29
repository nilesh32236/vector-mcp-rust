cat << 'INNER_EOF' > patch_watcher.patch
--- src/indexer/watcher.rs
+++ src/indexer/watcher.rs
@@ -10,7 +10,7 @@
 use crate::db::Store;
 use crate::indexer;
 use crate::llm::embedding::Embedder;
 use crate::llm::summarizer::Summarizer;

 /// Starts the background CDC file watcher.
 pub async fn start_watcher(
     config: Arc<Config>,
     db: Arc<Store>,
     embedder: Arc<Embedder>,
     summarizer: Arc<Summarizer>,
-) -> Result<()> {
+) -> Result<Option<notify::RecommendedWatcher>> {
     if !config.feature_toggles.enable_live_indexing {
         info!("Live indexing disabled via config");
-        return Ok(());
+        return Ok(None);
     }

     info!(
@@ -79,7 +79,5 @@
         }
     });

-    Box::leak(Box::new(watcher));
-
-    Ok(())
+    Ok(Some(watcher))
 }
INNER_EOF
patch src/indexer/watcher.rs < patch_watcher.patch
