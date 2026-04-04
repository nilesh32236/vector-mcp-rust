#![allow(dead_code)]
use crate::config::Config;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct GeminiMessage {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeminiPart {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct GeminiRequestContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
pub struct GeminiGenerateRequest {
    pub contents: Vec<GeminiRequestContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiSystemInstruction>,
}

#[derive(Debug, Serialize)]
pub struct GeminiSystemInstruction {
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiGenerateResponse {
    pub candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiCandidate {
    pub content: GeminiCandidateContent,
}

#[derive(Debug, Deserialize)]
pub struct GeminiCandidateContent {
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiModelList {
    pub models: Vec<GeminiModel>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiModel {
    pub name: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
    pub description: String,
}

pub struct GeminiClient {
    api_key: String,
    client: reqwest::Client,
}

impl GeminiClient {
    pub fn new(config: &Config) -> Self {
        Self {
            api_key: config.gemini_api_key.clone(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn generate_completion(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, self.api_key
        );

        let request = GeminiGenerateRequest {
            contents: vec![GeminiRequestContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: user_prompt.to_string(),
                }],
            }],
            system_instruction: if !system_prompt.is_empty() {
                Some(GeminiSystemInstruction {
                    parts: vec![GeminiPart {
                        text: system_prompt.to_string(),
                    }],
                })
            } else {
                None
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Gemini API")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Gemini API error ({}): {}",
                status,
                error_text
            ));
        }

        let result: GeminiGenerateResponse = response
            .json()
            .await
            .context("Failed to parse Gemini response")?;

        let text = result
            .candidates
            .first()
            .context("No candidates in Gemini response")?
            .content
            .parts
            .first()
            .context("No parts in Gemini candidate content")?
            .text
            .clone();

        Ok(text)
    }

    pub async fn list_models(&self) -> Result<Vec<GeminiModel>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models?key={}",
            self.api_key
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to list Gemini models")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Gemini API error ({}): {}",
                status,
                error_text
            ));
        }

        let result: GeminiModelList = response
            .json()
            .await
            .context("Failed to parse Gemini model list")?;
        Ok(result.models)
    }
}
