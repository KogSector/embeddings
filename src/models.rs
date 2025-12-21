//! API models for request/response types.

use serde::{Deserialize, Serialize};

/// Request for embedding a single text via API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    /// The text to embed.
    pub text: String,
    /// Source type for routing (e.g., "github", "notion", "code").
    #[serde(default)]
    pub source_type: Option<String>,
    /// Specific provider to use (optional, overrides routing).
    #[serde(default)]
    pub provider: Option<String>,
    /// Specific model to use (optional).
    #[serde(default)]
    pub model: Option<String>,
}

/// Response for a single embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Dimension of the embedding.
    pub dimension: u32,
    /// Model used for embedding.
    pub model: String,
    /// Provider used.
    pub provider: String,
}

/// Request for batch embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbedRequest {
    /// The texts to embed.
    pub texts: Vec<String>,
    /// Source type for routing.
    #[serde(default)]
    pub source_type: Option<String>,
    /// Specific provider to use.
    #[serde(default)]
    pub provider: Option<String>,
    /// Specific model to use.
    #[serde(default)]
    pub model: Option<String>,
}

/// Response for batch embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbedResponse {
    /// The embeddings for each input text.
    pub embeddings: Vec<Vec<f32>>,
    /// Dimension of the embeddings.
    pub dimension: u32,
    /// Model used.
    pub model: String,
    /// Provider used.
    pub provider: String,
    /// Number of texts processed.
    pub count: usize,
}

/// Request for embedding pre-chunked content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedChunksRequest {
    /// The chunks to embed.
    pub chunks: Vec<ChunkInput>,
    /// Source type for routing.
    #[serde(default)]
    pub source_type: Option<String>,
    /// Specific provider to use.
    #[serde(default)]
    pub provider: Option<String>,
}

/// A chunk of content to embed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInput {
    /// Unique identifier for the chunk.
    pub id: String,
    /// The text content of the chunk.
    pub content: String,
    /// Optional metadata about the chunk.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Response for chunk embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedChunksResponse {
    /// The embedded chunks.
    pub chunks: Vec<EmbeddedChunk>,
    /// Model used.
    pub model: String,
    /// Provider used.
    pub provider: String,
}

/// An embedded chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    /// The chunk ID.
    pub id: String,
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Dimension of the embedding.
    pub dimension: u32,
}

/// Request for reranking documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankApiRequest {
    /// The query to rank documents against.
    pub query: String,
    /// The documents to rerank.
    pub documents: Vec<String>,
    /// Number of top results to return.
    #[serde(default = "default_top_n")]
    pub top_n: usize,
    /// Optional model to use.
    #[serde(default)]
    pub model: Option<String>,
}

fn default_top_n() -> usize {
    10
}

/// Response for reranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankApiResponse {
    /// Reranked results.
    pub results: Vec<RerankResultItem>,
    /// Model used.
    pub model: String,
}

/// A single rerank result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResultItem {
    /// Original index in the input.
    pub index: usize,
    /// The document text.
    pub document: String,
    /// Relevance score (0-1).
    pub score: f32,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status.
    pub status: String,
    /// Service name.
    pub service: String,
    /// Service version.
    pub version: String,
    /// Available providers.
    pub providers: Vec<ProviderInfo>,
    /// Available endpoints.
    pub endpoints: Vec<String>,
}

/// Information about an available provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Provider name.
    pub name: String,
    /// Whether the provider is available.
    pub available: bool,
    /// Available models.
    pub models: Vec<String>,
}

/// Error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error message.
    pub error: String,
    /// Error code (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}
