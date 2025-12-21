//! HuggingFace embedding client.
//!
//! Supports HuggingFace Inference API and dedicated endpoints for sentence-transformers models.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Serialize;

use crate::traits::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingClient, EmbeddingRequest,
    EmbeddingResponse,
};

const HF_INFERENCE_API_BASE: &str = "https://api-inference.huggingface.co";

/// HuggingFace API embedding request.
#[derive(Debug, Serialize)]
struct HuggingFaceRequest {
    inputs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<HuggingFaceOptions>,
}

#[derive(Debug, Serialize)]
struct HuggingFaceOptions {
    wait_for_model: bool,
}

/// HuggingFace embedding client.
pub struct HuggingFaceClient {
    client: Client,
    api_token: String,
    base_url: String,
    dedicated_endpoint: Option<String>,
}

impl HuggingFaceClient {
    /// Create a new HuggingFace client with the given API token.
    pub fn new(api_token: String) -> Self {
        Self {
            client: Client::new(),
            api_token,
            base_url: HF_INFERENCE_API_BASE.to_string(),
            dedicated_endpoint: None,
        }
    }

    /// Create a new HuggingFace client with a custom base URL.
    pub fn with_base_url(api_token: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_token,
            base_url,
            dedicated_endpoint: None,
        }
    }

    /// Create a new HuggingFace client with a dedicated inference endpoint.
    pub fn with_endpoint(api_token: String, endpoint_url: String) -> Self {
        Self {
            client: Client::new(),
            api_token,
            base_url: HF_INFERENCE_API_BASE.to_string(),
            dedicated_endpoint: Some(endpoint_url),
        }
    }

    /// Get the URL for embedding requests.
    fn get_embedding_url(&self, model: &str) -> String {
        if let Some(ref endpoint) = self.dedicated_endpoint {
            endpoint.clone()
        } else {
            format!(
                "{}/pipeline/feature-extraction/{}",
                self.base_url, model
            )
        }
    }

    /// Generate embeddings for multiple texts.
    async fn generate_embeddings(
        &self,
        model: &str,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>> {
        let request = HuggingFaceRequest {
            inputs: texts,
            options: Some(HuggingFaceOptions {
                wait_for_model: true,
            }),
        };

        let url = self.get_embedding_url(model);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "HuggingFace API error ({}): {}",
                status,
                error_text
            ));
        }

        // HuggingFace can return different formats depending on the model
        // For sentence-transformers, it typically returns [[embedding1], [embedding2], ...]
        // Some models return [[[token_embeddings]]], so we need to handle mean pooling
        let raw_response: serde_json::Value = response.json().await?;

        let embeddings = self.parse_embeddings(raw_response)?;
        Ok(embeddings)
    }

    /// Parse embeddings from the HuggingFace response.
    fn parse_embeddings(&self, response: serde_json::Value) -> Result<Vec<Vec<f32>>> {
        match response {
            // Direct array of embeddings: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            serde_json::Value::Array(arr) => {
                let mut embeddings = Vec::new();
                for item in arr {
                    match item {
                        // Direct embedding vector
                        serde_json::Value::Array(embedding) if embedding.first().map_or(false, |v| v.is_f64()) => {
                            let emb: Vec<f32> = embedding
                                .into_iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect();
                            embeddings.push(emb);
                        }
                        // Token embeddings that need mean pooling: [[token1], [token2], ...]
                        serde_json::Value::Array(tokens) => {
                            let pooled = self.mean_pool_tokens(tokens)?;
                            embeddings.push(pooled);
                        }
                        _ => return Err(anyhow!("Unexpected embedding format")),
                    }
                }
                Ok(embeddings)
            }
            _ => Err(anyhow!("Unexpected response format from HuggingFace")),
        }
    }

    /// Mean pool token embeddings to get a single sentence embedding.
    fn mean_pool_tokens(&self, tokens: Vec<serde_json::Value>) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(anyhow!("No tokens to pool"));
        }

        let token_embeddings: Vec<Vec<f32>> = tokens
            .into_iter()
            .filter_map(|t| {
                t.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
            })
            .collect();

        if token_embeddings.is_empty() {
            return Err(anyhow!("Failed to parse token embeddings"));
        }

        let dim = token_embeddings[0].len();
        let num_tokens = token_embeddings.len() as f32;
        let mut pooled = vec![0.0f32; dim];

        for token in &token_embeddings {
            for (i, &val) in token.iter().enumerate() {
                if i < dim {
                    pooled[i] += val;
                }
            }
        }

        for val in &mut pooled {
            *val /= num_tokens;
        }

        Ok(pooled)
    }
}

#[async_trait]
impl EmbeddingClient for HuggingFaceClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let model = request
            .model
            .as_deref()
            .unwrap_or("sentence-transformers/all-MiniLM-L6-v2");

        let embeddings = self
            .generate_embeddings(model, vec![request.text])
            .await?;

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned from HuggingFace"))?;

        Ok(EmbeddingResponse {
            dimension: embedding.len() as u32,
            embedding,
            model: model.to_string(),
            usage: None,
        })
    }

    async fn embed_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse> {
        let model = request
            .model
            .as_deref()
            .unwrap_or("sentence-transformers/all-MiniLM-L6-v2");

        let embeddings = self.generate_embeddings(model, request.texts).await?;

        let dim = embeddings.first().map(|e| e.len() as u32).unwrap_or(384);

        Ok(BatchEmbeddingResponse {
            embeddings,
            dimension: dim,
            model: model.to_string(),
            usage: None,
        })
    }

    fn get_dimension(&self, model: &str) -> Option<u32> {
        match model {
            "sentence-transformers/all-MiniLM-L6-v2" => Some(384),
            "sentence-transformers/all-MiniLM-L12-v2" => Some(384),
            "sentence-transformers/all-mpnet-base-v2" => Some(768),
            "BAAI/bge-small-en-v1.5" => Some(384),
            "BAAI/bge-base-en-v1.5" => Some(768),
            "BAAI/bge-large-en-v1.5" => Some(1024),
            _ => None,
        }
    }

    fn provider_name(&self) -> &str {
        "huggingface"
    }

    fn default_model(&self) -> &str {
        "sentence-transformers/all-MiniLM-L6-v2"
    }

    fn is_available(&self) -> bool {
        !self.api_token.is_empty()
    }
}
