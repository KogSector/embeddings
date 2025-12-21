//! Jina AI embedding client.
//!
//! Supports jina-embeddings-v3 and jina-clip-v2.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::traits::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingClient, EmbeddingRequest,
    EmbeddingResponse, TokenUsage, RerankClient, RerankRequest, RerankResponse, RerankResult,
};

const JINA_API_BASE: &str = "https://api.jina.ai/v1";

/// Jina API embedding request.
#[derive(Debug, Serialize)]
struct JinaRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// Jina API embedding response.
#[derive(Debug, Deserialize)]
struct JinaResponse {
    model: String,
    object: String,
    data: Vec<JinaEmbeddingData>,
    usage: JinaUsage,
}

#[derive(Debug, Deserialize)]
struct JinaEmbeddingData {
    object: String,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct JinaUsage {
    total_tokens: usize,
    prompt_tokens: Option<usize>,
}

/// Jina rerank request.
#[derive(Debug, Serialize)]
struct JinaRerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: usize,
}

/// Jina rerank response.
#[derive(Debug, Deserialize)]
struct JinaRerankResponse {
    model: String,
    results: Vec<JinaRerankResult>,
    usage: JinaUsage,
}

#[derive(Debug, Deserialize)]
struct JinaRerankResult {
    index: usize,
    document: JinaDocument,
    relevance_score: f32,
}

#[derive(Debug, Deserialize)]
struct JinaDocument {
    text: String,
}

/// Jina AI embedding client.
pub struct JinaClient {
    client: Client,
    api_key: String,
}

impl JinaClient {
    /// Create a new Jina client with the given API key.
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
        task: Option<&str>,
        dimensions: Option<u32>,
    ) -> Result<(Vec<Vec<f32>>, JinaUsage)> {
        let request = JinaRequest {
            model: model.to_string(),
            input: texts,
            task: task.map(|s| s.to_string()),
            dimensions,
        };

        let response = self
            .client
            .post(format!("{}/embeddings", JINA_API_BASE))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Jina API error ({}): {}", status, error_text));
        }

        let api_response: JinaResponse = response.json().await?;

        // Sort by index to ensure correct order
        let mut data = api_response.data;
        data.sort_by_key(|d| d.index);

        let embeddings = data.into_iter().map(|d| d.embedding).collect();

        Ok((embeddings, api_response.usage))
    }
}

#[async_trait]
impl EmbeddingClient for JinaClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let model = request
            .model
            .as_deref()
            .unwrap_or("jina-embeddings-v3");
        let task = request.task_type.as_deref();

        let (embeddings, usage) = self
            .generate_embeddings(model, vec![request.text], task, request.output_dimension)
            .await?;

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned from Jina"))?;

        Ok(EmbeddingResponse {
            dimension: embedding.len() as u32,
            embedding,
            model: model.to_string(),
            usage: Some(TokenUsage {
                prompt_tokens: usage.prompt_tokens.unwrap_or(usage.total_tokens),
                total_tokens: usage.total_tokens,
            }),
        })
    }

    async fn embed_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse> {
        let model = request
            .model
            .as_deref()
            .unwrap_or("jina-embeddings-v3");
        let task = request.task_type.as_deref();

        let (embeddings, usage) = self
            .generate_embeddings(model, request.texts, task, request.output_dimension)
            .await?;

        let dim = embeddings.first().map(|e| e.len() as u32).unwrap_or(1024);

        Ok(BatchEmbeddingResponse {
            embeddings,
            dimension: dim,
            model: model.to_string(),
            usage: Some(TokenUsage {
                prompt_tokens: usage.prompt_tokens.unwrap_or(usage.total_tokens),
                total_tokens: usage.total_tokens,
            }),
        })
    }

    fn get_dimension(&self, model: &str) -> Option<u32> {
        match model {
            "jina-embeddings-v3" => Some(1024),
            "jina-embeddings-v2-base-en" => Some(768),
            "jina-embeddings-v2-small-en" => Some(512),
            "jina-clip-v2" => Some(1024),
            _ => Some(1024),
        }
    }

    fn provider_name(&self) -> &str {
        "jina"
    }

    fn default_model(&self) -> &str {
        "jina-embeddings-v3"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}

#[async_trait]
impl RerankClient for JinaClient {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse> {
        let model = request
            .model
            .as_deref()
            .unwrap_or("jina-reranker-v2-base-multilingual");

        let jina_request = JinaRerankRequest {
            model: model.to_string(),
            query: request.query,
            documents: request.documents,
            top_n: request.top_n,
        };

        let response = self
            .client
            .post(format!("{}/rerank", JINA_API_BASE))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&jina_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Jina Rerank API error ({}): {}", status, error_text));
        }

        let api_response: JinaRerankResponse = response.json().await?;

        let results = api_response
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                document: r.document.text,
                relevance_score: r.relevance_score,
            })
            .collect();

        Ok(RerankResponse {
            results,
            model: model.to_string(),
        })
    }

    fn provider_name(&self) -> &str {
        "jina"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}
