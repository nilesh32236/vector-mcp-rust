use anyhow::{anyhow, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use serde::Deserialize;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;
use std::io::{self, Write};

#[derive(Deserialize, Debug)]
struct Scenario {
    name: String,
    file: String,
    chunk: String,
    metadata: Metadata,
}

#[derive(Deserialize, Debug)]
struct Metadata {
    symbols: String,
    calls: String,
    context: String,
    relationships: String,
}

fn main() -> Result<()> {
    let mut model_path = PathBuf::new();
    let mut scenario_path = PathBuf::new();
    let mut output_path = None;

    // 1. CLI Argument Parsing
    let args: Vec<String> = std::env::args().collect();
    let mut iter = args.into_iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => {
                match iter.next() {
                    Some(val) => model_path = PathBuf::from(val),
                    None => {
                        eprintln!("Error: Missing value for --model");
                        std::process::exit(1);
                    }
                }
            }
            "--scenarios" => {
                match iter.next() {
                    Some(val) => scenario_path = PathBuf::from(val),
                    None => {
                        eprintln!("Error: Missing value for --scenarios");
                        std::process::exit(1);
                    }
                }
            }
            "--output" => {
                match iter.next() {
                    Some(val) => output_path = Some(PathBuf::from(val)),
                    None => {
                        eprintln!("Error: Missing value for --output");
                        std::process::exit(1);
                    }
                }
            }
            _ => {}
        }
    }

    if model_path.as_os_str().is_empty() || scenario_path.as_os_str().is_empty() {
        eprintln!("Usage: bench_summary --model <path> --scenarios <path.json> [--output <report.txt>]");
        std::process::exit(1);
    }

    // 2. Load Scenarios
    let scenarios_json = std::fs::read_to_string(&scenario_path)?;
    let scenarios: Vec<Scenario> = serde_json::from_str(&scenarios_json)?;

    // 3. Initialize Backend and Model
    let mut full_report = String::new();
    let line = "=========================================================";
    
    full_report.push_str(&format!("{}\nADVANCED ARCHITECTURAL SUMMARIZATION AUDIT\n", line));
    full_report.push_str(&format!("Model: {}\n", model_path.display()));
    full_report.push_str(&format!("{}\n\n", line));

    let backend = LlamaBackend::init().map_err(|e| anyhow!("Backend init failed: {e}"))?;
    // Allow n_gpu_layers to be configured via env var; default to 0 (CPU-only safe).
    let n_gpu_layers: u32 = std::env::var("N_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(n_gpu_layers)
        .with_use_mmap(true)
        .with_use_mlock(true);
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
    
    let n_ctx = 8192;
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // 4. Run each scenario
    for scenario in scenarios {
        println!("\n▶ Running Scenario: {}", scenario.name);
        full_report.push_str(&format!("### SCENARIO: {}\n", scenario.name));

        let prompt = format!(
            "<|im_start|>system\n\
            You are a highly technical senior software architect specializing in code documentation and semantic analysis.\n\
            Global Context: {}\n\
            <|im_end|>\n\
            <|im_start|>user\n\
            Analyze the following code chunk from `{}`:\n\
            \n\
            ```\n\
            {}\n\
            ```\n\
            \n\
            Architectural Metadata:\n\
            - Local Symbols: {}\n\
            - External Calls: {}\n\
            - Structural Relationship: {}\n\
            \n\
            Task: Write a precise doc-string style summary. \n\
            You MUST follow this exact format:\n\
            \"This [entity] is used in [file] for [purpose] with the following details: [analysis of calls and relationships].\"\n\
            <|im_end|>\n\
            <|im_start|>assistant\n",
            scenario.metadata.context,
            scenario.file,
            scenario.chunk,
            scenario.metadata.symbols,
            scenario.metadata.calls,
            scenario.metadata.relationships
        );

        let tokens = model.str_to_token(&prompt, AddBos::Always)?;
        if tokens.is_empty() {
            eprintln!("Warning: tokenization produced empty token list for scenario '{}' — skipping", scenario.name);
            continue;
        }
        let start_time = Instant::now();
        
        let n_batch = 512;
        let mut batch = LlamaBatch::new(n_batch, 1);
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(token, i as i32, &[0], is_last)?;
            if batch.n_tokens() as usize == n_batch || is_last {
                ctx.decode(&mut batch)?;
                if !is_last { batch.clear(); }
            }
        }
        let prefill_dur = start_time.elapsed();

        print!("> ");
        io::stdout().flush()?;
        
        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.1), LlamaSampler::greedy()]);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut ttft = None;
        let gen_start = Instant::now();
        let mut token_count: usize = 0;

        for i in 0..200_usize {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if ttft.is_none() { ttft = Some(gen_start.elapsed()); }
            if model.is_eog_token(token) { break; }
            sampler.accept(token);
            let piece = model.token_to_piece(token, &mut decoder, true, None)?;
            print!("{}", piece);
            io::stdout().flush()?;
            output.push_str(&piece);
            token_count += 1;

            batch.clear();
            let pos = tokens.len().saturating_add(i);
            let pos_i32 = pos.min(i32::MAX as usize) as i32;
            batch.add(token, pos_i32, &[0], true)?;
            ctx.decode(&mut batch)?;
        }

        let total_gen_time = gen_start.elapsed();
        let tps = if token_count == 0 || total_gen_time.is_zero() {
            0.0
        } else {
            token_count as f64 / total_gen_time.as_secs_f64()
        };

        full_report.push_str(&format!("- Summary: {}\n", output.trim()));
        full_report.push_str(&format!("- Metrics: TTFT: {:?}, TPS: {:.2}, Prefill: {:?}\n\n", ttft.unwrap_or_default(), tps, prefill_dur));
        println!("\n✓ Metrics: TTFT: {:?}, TPS: {:.2}", ttft.unwrap_or_default(), tps);
        
        // Reset context for next scenario
        ctx.clear_kv_cache();
    }

    if let Some(path) = output_path {
        std::fs::write(&path, &full_report)?;
        println!("\n✅ Full Audit Report saved to: {}", path.display());
    }

    Ok(())
}
