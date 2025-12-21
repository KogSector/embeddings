//! Voyage AI embedding client.
//!
//! Supports voyage-3, voyage-3-large, and voyage-code-3.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::traits::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingClient, EmbeddingRequest,
    EmbeddingResponse, TokenUsage,
};

const VOYAGE_API_BASE: &str = "https://api.voyageai.com/v1";

/// Voyage API embedding request.
#[derive(Debug, Serialize)]
struct VoyageRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<bool>,
}

/// Voyage API embedding response.
#[derive(Debug, Deserialize)]
struct VoyageResponse {
    object: String,
    data: Vec<VoyageEmbeddingData>,
    model: String,
    usage: VoyageUsage,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbeddingData {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct VoyageUsage {
    total_tokens: usize,
}

/// Voyage AI embedding client.
pub struct VoyageClient {
    client: Client,
    api_key: String,
}

impl VoyageClient {
    /// Create a new Voyage client with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    /// Generate embeddings for multiple texts.
    async fn generate_embeddings(
        &self,
        model: &str,
        texts: Vec<String>,
        input_type: Option<&str>,
    ) -> Result<(Vec<Vec<f32>>, usize)> {
        let request = VoyageRequest {
            model: model.to_string(),
            input: texts,
            input_type: input_type.map(|s| s.to_string()),
            truncation: Some(true),
        };

        let response = self
            .client
            .post(format!("{}/embeddings", VOYAGE_API_BASE))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Voyage API error ({}): {}", status, error_text));
        }

        let api_response: VoyageResponse = response.json().await?;

        // Sort by index to ensure correct order
        let mut data = api_response.data;
        data.sort_by_key(|d| d.index);

        let embeddings = data.into_iter().map(|d| d.embedding).collect();

        Ok((embeddings, api_response.usage.total_tokens))
    }
}

#[async_trait]
impl EmbeddingClient for VoyageClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let model = request.model.as_deref().unwrap_or("voyage-3");
        let input_type = request.task_type.as_deref();

        let (embeddings, tokens) = self
            .generate_embeddings(model, vec![request.text], input_type)
            .await?;

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned from Voyage"))?;

        Ok(EmbeddingResponse {
            dimension: embedding.len() as u32,
            embedding,
            model: model.to_string(),
            usage: Some(TokenUsage {
                prompt_tokens: tokens,
                total_tokens: tokens,
            }),
        })
    }

    async fn embed_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse> {
        let model = request.model.as_deref().unwrap_or("voyage-3");
        let input_type = request.task_type.as_deref();

        let (embeddings, tokens) = self
            .generate_embeddings(model, request.texts, input_type)
            .await?;

        let dim = embeddings.first().map(|e| e.len() as u32).unwrap_or(1024);

        Ok(BatchEmbeddingResponse {
            embeddings,
            dimension: dim,
            model: model.to_string(),
            usage: Some(TokenUsage {
                prompt_tokens: tokens,
                total_tokens: tokens,
            }),
        })
    }

    fn get_dimension(&self, model: &str) -> Option<u32> {
        match model {
            "voyage-3" => Some(1024),
            "voyage-3-large" => Some(1024),
            "voyage-3-lite" => Some(512),
            "voyage-code-3" => Some(1024),
            "voyage-finance-2" => Some(1024),
            "voyage-law-2" => Some(1024),
            _ => Some(1024),
        }
    }

    fn provider_name(&self) -> &str {
        "voyage"
    }

    fn default_model(&self) -> &str {
        "voyage-3"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}
