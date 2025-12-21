//! OpenAI embedding client.
//!
//! Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::traits::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingClient, EmbeddingRequest,
    EmbeddingResponse, TokenUsage,
};

const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// OpenAI API embedding request.
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// OpenAI API embedding response.
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    object: String,
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// OpenAI embedding client.
pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIClient {
    /// Create a new OpenAI client with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: OPENAI_API_BASE.to_string(),
        }
    }

    /// Create a new OpenAI client with a custom base URL.
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }

    /// Generate embeddings for multiple texts.
    async fn generate_embeddings(
        &self,
        model: &str,
        texts: Vec<String>,
        dimensions: Option<u32>,
    ) -> Result<(Vec<Vec<f32>>, OpenAIUsage)> {
        let request = OpenAIRequest {
            model: model.to_string(),
            input: texts,
            dimensions,
        };

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("OpenAI API error ({}): {}", status, error_text));
        }

        let api_response: OpenAIResponse = response.json().await?;

        // Sort by index to ensure correct order
        let mut data = api_response.data;
        data.sort_by_key(|d| d.index);

        let embeddings = data.into_iter().map(|d| d.embedding).collect();

        Ok((embeddings, api_response.usage))
    }
}

#[async_trait]
impl EmbeddingClient for OpenAIClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let model = request.model.as_deref().unwrap_or("text-embedding-3-small");
        let dimension = request.output_dimension.or_else(|| self.get_dimension(model));

        let (embeddings, usage) = self
            .generate_embeddings(model, vec![request.text], dimension)
            .await?;

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned from OpenAI"))?;

        Ok(EmbeddingResponse {
            dimension: embedding.len() as u32,
            embedding,
            model: model.to_string(),
            usage: Some(TokenUsage {
                prompt_tokens: usage.prompt_tokens,
                total_tokens: usage.total_tokens,
            }),
        })
    }

    async fn embed_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse> {
        let model = request
            .model
            .as_deref()
            .unwrap_or("text-embedding-3-small");
        let dimension = request.output_dimension.or_else(|| self.get_dimension(model));

        let (embeddings, usage) = self
            .generate_embeddings(model, request.texts, dimension)
            .await?;

        let dim = embeddings.first().map(|e| e.len() as u32).unwrap_or(1536);

        Ok(BatchEmbeddingResponse {
            embeddings,
            dimension: dim,
            model: model.to_string(),
            usage: Some(TokenUsage {
                prompt_tokens: usage.prompt_tokens,
                total_tokens: usage.total_tokens,
            }),
        })
    }

    fn get_dimension(&self, model: &str) -> Option<u32> {
        match model {
            "text-embedding-3-large" => Some(3072),
            "text-embedding-3-small" => Some(1536),
            "text-embedding-ada-002" => Some(1536),
            _ => Some(1536),
        }
    }

    fn provider_name(&self) -> &str {
        "openai"
    }

    fn default_model(&self) -> &str {
        "text-embedding-3-small"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_lookup() {
        let client = OpenAIClient::new("test".to_string());
        assert_eq!(client.get_dimension("text-embedding-3-small"), Some(1536));
        assert_eq!(client.get_dimension("text-embedding-3-large"), Some(3072));
        assert_eq!(client.get_dimension("text-embedding-ada-002"), Some(1536));
    }
}
