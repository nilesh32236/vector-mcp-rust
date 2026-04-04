use anyhow::{Context, Result};
use tracing::info;

pub fn download_direct(url: &str, dest: &std::path::Path) -> Result<()> {
    info!("Direct downloading {} to {:?}", url, dest);
    let mut response = reqwest::blocking::get(url).context("Failed to GET URL")?;
    if !response.status().is_success() {
        return Err(anyhow::anyhow!("Failed to download {}: status {}", url, response.status()));
    }
    let mut out = std::fs::File::create(dest).context("Failed to create destination file")?;
    std::io::copy(&mut response, &mut out).context("Failed to copy response to file")?;
    Ok(())
}
