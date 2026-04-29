use anyhow::{Context, Result};
use tracing::info;

pub async fn download_direct(url: &str, dest: &std::path::Path) -> Result<()> {
    info!("Direct downloading {} to {:?}", url, dest);
    let response = reqwest::get(url).await.context("Failed to GET URL")?;
    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to download {}: status {}",
            url,
            response.status()
        ));
    }
    let content = response.bytes().await.context("Failed to read response bytes")?;
    std::fs::write(dest, content).context("Failed to write destination file")?;
    Ok(())
}
