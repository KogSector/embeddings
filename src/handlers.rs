//! HTTP handlers module.

use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use std::sync::Arc;
use tracing::{info, error};

use crate::config::Config;
use crate::models::{
    EmbedRequest, EmbedResponse, BatchEmbedRequest, BatchEmbedResponse,
    EmbedChunksRequest, EmbedChunksResponse, EmbeddedChunk,
    RerankApiRequest, RerankApiResponse, RerankResultItem,
    HealthResponse, ProviderInfo, ErrorResponse,
};
use crate::services::EmbeddingOrchestrator;
use crate::traits::RerankRequest;

/// Application state shared across handlers.
pub struct AppState {
    pub orchestrator: Arc<EmbeddingOrchestrator>,
    pub config: Config,
}

/// Health check endpoint.
pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    let providers: Vec<ProviderInfo> = state.orchestrator.available_providers()
        .into_iter()
        .map(|name| ProviderInfo {
            name: name.clone(),
            available: true,
            models: get_models_for_provider(&name),
        })
        .collect();

    Json(HealthResponse {
        status: "healthy".to_string(),
        service: "embeddings".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        providers,
        endpoints: vec![
            "/health".to_string(),
            "/embed".to_string(),
            "/batch/embed".to_string(),
            "/batch/embed/chunks".to_string(),
            "/rerank".to_string(),
        ],
    })
}

/// Get available models for a provider.
fn get_models_for_provider(provider: &str) -> Vec<String> {
    match provider {
        "openai" => vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
        ],
        "cohere" => vec![
            "embed-english-v3.0".to_string(),
            "embed-multilingual-v3.0".to_string(),
        ],
        "voyage" => vec![
            "voyage-3".to_string(),
            "voyage-3-large".to_string(),
            "voyage-code-3".to_string(),
        ],
        "jina" => vec![
            "jina-embeddings-v3".to_string(),
        ],
        "huggingface" => vec![
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            "BAAI/bge-large-en-v1.5".to_string(),
        ],
        _ => vec![],
    }
}

/// Embed a single text.
pub async fn embed_single(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("Embedding single text, source_type: {:?}", request.source_type);

    match state.orchestrator.embed(
        request.text,
        request.source_type.as_deref(),
        request.provider.as_deref(),
        request.model,
    ).await {
        Ok(response) => {
            let provider = request.provider.unwrap_or_else(|| state.config.default_provider.clone());
            Ok(Json(EmbedResponse {
                embedding: response.embedding,
                dimension: response.dimension,
                model: response.model,
                provider,
            }))
        }
        Err(e) => {
            error!("Embedding failed: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: Some("EMBEDDING_FAILED".to_string()),
                }),
            ))
        }
    }
}

/// Embed multiple texts in batch.
pub async fn embed_batch(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchEmbedRequest>,
) -> Result<Json<BatchEmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("Embedding batch of {} texts, source_type: {:?}", request.texts.len(), request.source_type);

    if request.texts.is_empty() {
        return Ok(Json(BatchEmbedResponse {
            embeddings: vec![],
            dimension: 0,
            model: String::new(),
            provider: state.config.default_provider.clone(),
            count: 0,
        }));
    }

    match state.orchestrator.embed_batch(
        request.texts.clone(),
        request.source_type.as_deref(),
        request.provider.as_deref(),
        request.model,
    ).await {
        Ok(response) => {
            let provider = request.provider.unwrap_or_else(|| state.config.default_provider.clone());
            Ok(Json(BatchEmbedResponse {
                count: response.embeddings.len(),
                embeddings: response.embeddings,
                dimension: response.dimension,
                model: response.model,
                provider,
            }))
        }
        Err(e) => {
            error!("Batch embedding failed: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: Some("BATCH_EMBEDDING_FAILED".to_string()),
                }),
            ))
        }
    }
}

/// Embed pre-chunked content.
pub async fn embed_chunks(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbedChunksRequest>,
) -> Result<Json<EmbedChunksResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("Embedding {} chunks, source_type: {:?}", request.chunks.len(), request.source_type);

    if request.chunks.is_empty() {
        return Ok(Json(EmbedChunksResponse {
            chunks: vec![],
            model: String::new(),
            provider: state.config.default_provider.clone(),
        }));
    }

    // Extract texts from chunks
    let texts: Vec<String> = request.chunks.iter().map(|c| c.content.clone()).collect();
    let ids: Vec<String> = request.chunks.iter().map(|c| c.id.clone()).collect();

    match state.orchestrator.embed_batch(
        texts,
        request.source_type.as_deref(),
        request.provider.as_deref(),
        None,
    ).await {
        Ok(response) => {
            let embedded_chunks: Vec<EmbeddedChunk> = ids
                .into_iter()
                .zip(response.embeddings.into_iter())
                .map(|(id, embedding)| EmbeddedChunk {
                    id,
                    dimension: embedding.len() as u32,
                    embedding,
                })
                .collect();

            let provider = request.provider.unwrap_or_else(|| state.config.default_provider.clone());

            Ok(Json(EmbedChunksResponse {
                chunks: embedded_chunks,
                model: response.model,
                provider,
            }))
        }
        Err(e) => {
            error!("Chunk embedding failed: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: Some("CHUNK_EMBEDDING_FAILED".to_string()),
                }),
            ))
        }
    }
}

/// Rerank documents by relevance.
pub async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RerankApiRequest>,
) -> Result<Json<RerankApiResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("Reranking {} documents", request.documents.len());

    let rerank_request = RerankRequest {
        query: request.query,
        documents: request.documents,
        top_n: request.top_n,
        model: request.model,
    };

    match state.orchestrator.rerank(rerank_request).await {
        Ok(response) => {
            let results: Vec<RerankResultItem> = response.results
                .into_iter()
                .map(|r| RerankResultItem {
                    index: r.index,
                    document: r.document,
                    score: r.relevance_score,
                })
                .collect();

            Ok(Json(RerankApiResponse {
                results,
                model: response.model,
            }))
        }
        Err(e) => {
            error!("Reranking failed: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: Some("RERANK_FAILED".to_string()),
                }),
            ))
        }
    }
}
