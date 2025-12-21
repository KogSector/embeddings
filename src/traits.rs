//! Core traits for embedding clients.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Request for generating an embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// The text to embed.
    pub text: String,
    /// The model to use (provider-specific).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional output dimension (for models that support it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<u32>,
    /// Task type hint (e.g., "search_document", "search_query").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
}

/// Response containing the generated embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// The dimension of the embedding.
    pub dimension: u32,
    /// The model used to generate the embedding.
    pub model: String,
    /// Token usage information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// Batch embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingRequest {
    /// The texts to embed.
    pub texts: Vec<String>,
    /// The model to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional output dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<u32>,
    /// Task type hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
}

/// Batch embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingResponse {
    /// The embeddings for each input text.
    pub embeddings: Vec<Vec<f32>>,
    /// The dimension of the embeddings.
    pub dimension: u32,
    /// The model used.
    pub model: String,
    /// Total token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

/// Rerank request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    /// The query to rank documents against.
    pub query: String,
    /// The documents to rerank.
    pub documents: Vec<String>,
    /// Number of top results to return.
    #[serde(default = "default_top_n")]
    pub top_n: usize,
    /// The model to use for reranking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

fn default_top_n() -> usize {
    10
}

/// Rerank response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Reranked results with scores.
    pub results: Vec<RerankResult>,
    /// The model used.
    pub model: String,
}

/// A single rerank result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Original index in the input documents.
    pub index: usize,
    /// The document text.
    pub document: String,
    /// Relevance score.
    pub relevance_score: f32,
}

/// Trait for embedding clients that can generate embeddings.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Generate an embedding for a single text.
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Generate embeddings for multiple texts.
    async fn embed_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse>;

    /// Get the default embedding dimension for a model.
    fn get_dimension(&self, model: &str) -> Option<u32>;

    /// Get the provider name.
    fn provider_name(&self) -> &str;

    /// Get the default model for this provider.
    fn default_model(&self) -> &str;

    /// Check if the client is properly configured.
    fn is_available(&self) -> bool;
}

/// Trait for reranking clients.
#[async_trait]
pub trait RerankClient: Send + Sync {
    /// Rerank documents by relevance to a query.
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse>;

    /// Get the provider name.
    fn provider_name(&self) -> &str;

    /// Check if reranking is available.
    fn is_available(&self) -> bool;
}
