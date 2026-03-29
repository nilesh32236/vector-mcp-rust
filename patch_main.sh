cat << 'INNER_EOF' > patch_main.patch
--- src/main.rs
+++ src/main.rs
@@ -32,7 +32,7 @@
     store: Arc<Store>,
     embedder: Arc<Embedder>,
     summarizer: Arc<Summarizer>,
-) -> Result<()> {
+) -> Result<Option<notify::RecommendedWatcher>> {
     let config_clone = Arc::clone(&config);
     let store_clone = Arc::clone(&store);
     let embedder_clone = Arc::clone(&embedder);
@@ -48,14 +48,14 @@
         .await;
     });

-    indexer::watcher::start_watcher(
+    let watcher = indexer::watcher::start_watcher(
         Arc::clone(&config),
         Arc::clone(&store),
         Arc::clone(&embedder),
         Arc::clone(&summarizer),
     )
     .await
     .context("Starting background watcher")?;

-    Ok(())
+    Ok(watcher)
 }

 #[tokio::main]
@@ -79,7 +79,7 @@
     let config = Arc::new(cfg);

     // 5. Initial Scan & Background Watcher (if enabled).
-    start_background_tasks(
+    let _watcher = start_background_tasks(
         Arc::clone(&config),
         Arc::clone(&store),
         Arc::clone(&embedder),
INNER_EOF
patch src/main.rs < patch_main.patch
