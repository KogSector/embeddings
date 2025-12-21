//! Cohere embedding client.
//!
//! Supports embed-english-v3.0 and embed-multilingual-v3.0.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::traits::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingClient, EmbeddingRequest,
    EmbeddingResponse, TokenUsage,
};

const COHERE_API_BASE: &str = "https://api.cohere.ai/v1";

/// Input type for Cohere embeddings.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CohereInputType {
    SearchDocument,
    SearchQuery,
    Classification,
    Clustering,
}

/// Cohere API embedding request.
#[derive(Debug, Serialize)]
struct CohereRequest {
    model: String,
    texts: Vec<String>,
    input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<String>,
}

/// Cohere API embedding response.
#[derive(Debug, Deserialize)]
struct CohereResponse {
    id: String,
    embeddings: Vec<Vec<f32>>,
    texts: Vec<String>,
    meta: CohereMeta,
}

#[derive(Debug, Deserialize)]
struct CohereMeta {
    api_version: CohereApiVersion,
    billed_units: Option<CohereBilledUnits>,
}

#[derive(Debug, Deserialize)]
struct CohereApiVersion {
    version: String,
}

#[derive(Debug, Deserialize)]
struct CohereBilledUnits {
    input_tokens: Option<usize>,
}

/// Cohere embedding client.
pub struct CohereClient {
    client: Client,
    api_key: String,
}

impl CohereClient {
    /// Create a new Cohere client with the given API key.
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
        input_type: &str,
    ) -> Result<(Vec<Vec<f32>>, Option<usize>)> {
        let request = CohereRequest {
            model: model.to_string(),
            texts,
            input_type: input_type.to_string(),
            truncate: Some("END".to_string()),
        };

        let response = self
            .client
            .post(format!("{}/embed", COHERE_API_BASE))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Cohere API error ({}): {}", status, error_text));
        }

        let api_response: CohereResponse = response.json().await?;
        let tokens = api_response
            .meta
            .billed_units
            .and_then(|b| b.input_tokens);

        Ok((api_response.embeddings, tokens))
    }
}

#[async_trait]
impl EmbeddingClient for CohereClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let model = request
            .model
            .as_deref()
            .unwrap_or("embed-english-v3.0");
        let input_type = request
            .task_type
            .as_deref()
            .unwrap_or("search_document");

        let (embeddings, tokens) = self
            .generate_embeddings(model, vec![request.text], input_type)
            .await?;

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned from Cohere"))?;

        Ok(EmbeddingResponse {
            dimension: embedding.len() as u32,
            embedding,
            model: model.to_string(),
            usage: tokens.map(|t| TokenUsage {
                prompt_tokens: t,
                total_tokens: t,
            }),
        })
    }

    async fn embed_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse> {
        let model = request
            .model
            .as_deref()
            .unwrap_or("embed-english-v3.0");
        let input_type = request
            .task_type
            .as_deref()
            .unwrap_or("search_document");

        let (embeddings, tokens) = self
            .generate_embeddings(model, request.texts, input_type)
            .await?;

        let dim = embeddings.first().map(|e| e.len() as u32).unwrap_or(1024);

        Ok(BatchEmbeddingResponse {
            embeddings,
            dimension: dim,
            model: model.to_string(),
            usage: tokens.map(|t| TokenUsage {
                prompt_tokens: t,
                total_tokens: t,
            }),
        })
    }

    fn get_dimension(&self, model: &str) -> Option<u32> {
        match model {
            "embed-english-v3.0" => Some(1024),
            "embed-multilingual-v3.0" => Some(1024),
            "embed-english-light-v3.0" => Some(384),
            "embed-multilingual-light-v3.0" => Some(384),
            _ => Some(1024),
        }
    }

    fn provider_name(&self) -> &str {
        "cohere"
    }

    fn default_model(&self) -> &str {
        "embed-english-v3.0"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}
