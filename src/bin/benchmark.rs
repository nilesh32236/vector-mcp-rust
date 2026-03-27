use anyhow::Result;
use std::time::Instant;
use tokio::fs;

#[tokio::main]
async fn main() -> Result<()> {
    let path = "dummy_large_file.txt";
    // 50 MB
    let size = 50 * 1024 * 1024;
    let dummy_data = vec![0u8; size];
    fs::write(path, &dummy_data).await?;

    let iterations = 100;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::fs::read(path).unwrap();
    }
    let duration_sync = start.elapsed();
    println!("Sync Read Duration: {:?}", duration_sync);

    let start_async = Instant::now();
    for _ in 0..iterations {
        let _ = tokio::fs::read(path).await.unwrap();
    }
    let duration_async = start_async.elapsed();
    println!("Async Read Duration: {:?}", duration_async);

    fs::remove_file(path).await?;

    Ok(())
}
