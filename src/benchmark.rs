use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Instant;
use crate::indexer::chunker;
use crate::llm::models::LlamaEngine;

pub fn run(args: Vec<String>) -> Result<()> {
    let mut model_path = PathBuf::new();
    let mut test_file = PathBuf::new();
    let mut output_path = None;
    
    // Parse args
    let mut iter = args.into_iter().skip(1);
    while let Some(arg) = iter.next() {
        if arg == "benchmark" { continue; }
        match arg.as_str() {
            "--model" => {
                if let Some(val) = iter.next() {
                    model_path = PathBuf::from(val);
                }
            }
            "--test-file" => {
                if let Some(val) = iter.next() {
                    test_file = PathBuf::from(val);
                }
            }
            "--output" => {
                if let Some(val) = iter.next() {
                    output_path = Some(PathBuf::from(val));
                }
            }
            _ => {}
        }
    }

    if model_path.as_os_str().is_empty() || test_file.as_os_str().is_empty() {
        eprintln!("Usage: benchmark --model <path_to_gguf> --test-file <path_to_file>");
        std::process::exit(1);
    }

    if !model_path.exists() {
        anyhow::bail!("Model file not found: {}", model_path.display());
    }
    if !test_file.exists() {
        anyhow::bail!("Test file not found: {}", test_file.display());
    }

    let mut report = String::new();
    let line = "=========================================================";
    
    report.push_str(&format!("{}\nEmbedding Benchmark Report\n", line));
    report.push_str(&format!("Model: {}\n", model_path.display()));
    report.push_str(&format!("Test File: {}\n", test_file.display()));
    report.push_str(&format!("{}\n\n", line));

    println!("{}", report);

    let models_dir = model_path.parent().unwrap_or_else(|| std::path::Path::new("."));

    println!("[1/5] Initialising LlamaEngine...");
    let start_init = Instant::now();
    let engine = LlamaEngine::new_with_config(
        models_dir,
        None,
        None,
        None,
        Some(&model_path),
    )?;
    let init_dur = start_init.elapsed();
    println!("✓ Engine initialised in {:?}", init_dur);
    report.push_str(&format!("Initialization Time: {:?}\n\n", init_dur));
    
    println!("      Model loaded into memory. Check your peak RAM usage now!");
    println!("      Resuming in 3 seconds...\n");
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Chunking the test file
    println!("[2/5] Parsing Test File...");
    let content = std::fs::read_to_string(&test_file)?;
    let extension = format!(".{}", test_file.extension().unwrap().to_str().unwrap());
    let mut chunks = chunker::parse_file(&content, test_file.to_str().unwrap(), &extension)?;
    
    if chunks.is_empty() {
        chunks.push(chunker::Chunk {
            content: content.clone(),
            contextual_string: String::new(),
            symbols: vec![],
            relationships: vec![],
            node_type: "text".to_string(),
            calls: vec![],
            callee_relationships: vec![],
            impl_relationships: vec![],
            extends_relationships: vec![],
            function_score: 1.0,
            start_line: 1,
            end_line: content.lines().count(),
        });
    }

    chunks.truncate(50);
    println!("✓ Extracted {} chunks.\n", chunks.len());
    report.push_str(&format!("Total Chunks: {}\n\n", chunks.len()));

    // Phase 1: Speed Tests
    println!("[3/5] Speed (Latency & Throughput)");
    report.push_str("[Speed Results]\n");
    
    // Cold Start
    let cold_start = Instant::now();
    let _ = engine.generate_embedding(&chunks[0].content)?;
    let cold_dur = cold_start.elapsed();
    println!("- Cold Start Latency (1st chunk): {:?}", cold_dur);
    report.push_str(&format!("Cold Start (1st chunk): {:?}\n", cold_dur));

    // Warm Single Latency
    let warm_start = Instant::now();
    let _ = engine.generate_embedding(&chunks[0].content)?;
    let warm_dur = warm_start.elapsed();
    println!("- Warm Single Latency (2nd chunk): {:?}", warm_dur);
    report.push_str(&format!("Warm Single Latency:   {:?}\n", warm_dur));

    // Batch Throughput
    println!("- Running Batch Throughput ({} chunks)...", chunks.len());
    let mut embedded_chunks = Vec::new();
    let batch_start = Instant::now();
    for chunk in &chunks {
        let vec = engine.generate_embedding(&chunk.content)?;
        embedded_chunks.push((chunk.clone(), vec));
    }
    let batch_time = batch_start.elapsed();
    let avg_time = batch_time.as_millis() as f64 / chunks.len() as f64;
    println!("✓ Batch Throughput Time: {:?}", batch_time);
    println!("✓ Average Time / Chunk:  {:.2} ms\n", avg_time);
    report.push_str(&format!("Batch Total Time:      {:?}\n", batch_time));
    report.push_str(&format!("Average Time/Chunk:    {:.2} ms\n\n", avg_time));

    // Phase 3: Semantic Accuracy (Vibe Test)
    println!("[4/5] Semantic Accuracy (Vibe Test)");
    report.push_str("[Vibe Test Results]\n");
    let test_queries = vec![
        "Where is the Tantivy BM25 indexing logic?",
        "How are Tree-sitter semantic chunks extracted from files?",
        "How does the LlamaEngine generate Vulkan embeddings?",
    ];

    for query in test_queries {
        println!("---------------------------------------------------------");
        println!("Query: \"{}\"", query);
        report.push_str(&format!("Query: \"{}\"\n", query));
        let query_vec = engine.generate_embedding(query)?;
        
        let mut scores: Vec<(f32, &chunker::Chunk)> = embedded_chunks.iter().map(|(chunk, emb)| {
            let score = cosine_similarity(&query_vec, emb);
            (score, chunk)
        }).collect();

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for (i, (score, chunk)) in scores.iter().take(3).enumerate() {
            let preview: String = chunk.content.lines().take(3).collect::<Vec<_>>().join("\n");
            let output = format!("  #{}: Score {:.4} | Lines {}-{} | Preview: {}...", i+1, score, chunk.start_line, chunk.end_line, preview.replace("\n", " "));
            println!("{}", output);
            report.push_str(&format!("{}\n", output));
        }
        report.push_str("\n");
    }

    // Phase 5: Ground Truth Audit (Golden Standard)
    println!("[5/5] Ground Truth Audit (Accuracy Scoring)");
    report.push_str("[Ground Truth Accuracy]\n");
    
    let mut total_points = 0.0;
    let mut possible_points = 0.0;

    let golden_tests = vec![
        ("How is distance between points calculated?", "math_utils.go", "math.Sqrt"),
        ("How to load configuration from JSON?", "config_manager.py", "json.load"),
        ("How are Tree-sitter chunks extracted?", "chunker.rs", "fn tree_sitter_chunk"),
        ("Where is email validation logic?", "config_manager.py", "def validate_email"),
    ];

    for (query, file_name, marker) in golden_tests {
        // Only run if the file being tested matches or if we're doing a general audit
        if !test_file.to_str().unwrap().contains(file_name) {
            continue;
        }

        println!("Audit: \"{}\"", query);
        possible_points += 1.0;
        
        let query_vec = engine.generate_embedding(query)?;
        let mut scores: Vec<(f32, &chunker::Chunk)> = embedded_chunks.iter().map(|(chunk, emb)| {
            let score = cosine_similarity(&query_vec, emb);
            (score, chunk)
        }).collect();
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Check Top-3
        let mut found_rank = None;
        for (i, (_score, chunk)) in scores.iter().take(3).enumerate() {
            if chunk.content.contains(marker) {
                found_rank = Some(i + 1);
                break;
            }
        }

        match found_rank {
            Some(1) => {
                println!("  ✅ Perfect Hit (#1)");
                total_points += 1.0;
                report.push_str(&format!("Query: \"{}\" -> ✅ Match #1 (100%)\n", query));
            }
            Some(r) => {
                println!("  ⚠️ Partial Match (#{}/3)", r);
                total_points += 0.5;
                report.push_str(&format!("Query: \"{}\" -> ⚠️ Match #{} (50%)\n", query, r));
            }
            None => {
                println!("  ❌ Missed");
                report.push_str(&format!("Query: \"{}\" -> ❌ Miss (0%)\n", query));
            }
        }
    }

    if possible_points > 0.0 {
        let accuracy = (total_points / possible_points) * 100.0;
        println!("\nFinal Accuracy Score: {:.1}%\n", accuracy);
        report.push_str(&format!("\nFINAL ACCURACY SCORE: {:.1}%\n\n", accuracy));
    }
    
    let footer = "=========================================================";
    println!("{}", footer);
    println!("Benchmark Complete!");
    if let Some(path) = output_path {
        std::fs::write(&path, &report)?;
        println!("✓ Detailed report saved to: {}", path.display());
    }
    println!("{}", footer);

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot
}
