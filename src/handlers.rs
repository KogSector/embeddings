//! HTTP handlers module.
//!
//! Provides HTTP endpoints for embedding operations using the local model.

use axum::{extract::State, http::StatusCode, Json};
use std::sync::Arc;
use tracing::{error, info};

use crate::config::Config;
use crate::models::{
    BatchEmbedRequest, BatchEmbedResponse, EmbedChunksRequest, EmbedChunksResponse,
    EmbedRequest, EmbedResponse, EmbeddedChunk, ErrorResponse, HealthResponse, ProviderInfo,
};
use crate::services::EmbeddingOrchestrator;

/// Application state shared across handlers.
pub struct AppState {
    pub orchestrator: Arc<EmbeddingOrchestrator>,
    pub config: Config,
}

/// Health check endpoint.
pub async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let providers: Vec<ProviderInfo> = state
        .orchestrator
        .available_providers()
        .into_iter()
        .map(|name| ProviderInfo {
            name: name.clone(),
            available: true,
            models: vec![state.config.model_name.clone()],
        })
        .collect();

    let (cache_hits, cache_size) = state.orchestrator.cache_stats();

    Json(HealthResponse {
        status: "healthy".to_string(),
        service: "embeddings".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: state.config.model_name.clone(),
        dimension: state.config.model_dimension,
        providers,
        cache_stats: Some(CacheStats {
            hits: cache_hits,
            size: cache_size,
        }),
        endpoints: vec![
            "/health".to_string(),
            "/embed".to_string(),
            "/batch/embed".to_string(),
            "/batch/embed/chunks".to_string(),
        ],
    })
}

/// Cache statistics for health check.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub hits: usize,
    pub size: usize,
}

/// Embed a single text.
pub async fn embed_single(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!(
        "Embedding single text, source_type: {:?}",
        request.source_type
    );

    match state
        .orchestrator
        .embed(
            request.text,
            request.source_type.as_deref(),
            None, // provider (always local)
            None, // model (always default)
        )
        .await
    {
        Ok(response) => Ok(Json(EmbedResponse {
            embedding: response.embedding,
            dimension: response.dimension,
            model: response.model,
            provider: "local".to_string(),
        })),
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
    info!(
        "Embedding batch of {} texts, source_type: {:?}",
        request.texts.len(),
        request.source_type
    );

    if request.texts.is_empty() {
        return Ok(Json(BatchEmbedResponse {
            embeddings: vec![],
            dimension: state.config.model_dimension,
            model: state.config.model_name.clone(),
            provider: "local".to_string(),
            count: 0,
        }));
    }

    match state
        .orchestrator
        .embed_batch(
            request.texts.clone(),
            request.source_type.as_deref(),
            None,
            None,
        )
        .await
    {
        Ok(response) => Ok(Json(BatchEmbedResponse {
            count: response.embeddings.len(),
            embeddings: response.embeddings,
            dimension: response.dimension,
            model: response.model,
            provider: "local".to_string(),
        })),
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
    info!(
        "Embedding {} chunks, source_type: {:?}",
        request.chunks.len(),
        request.source_type
    );

    if request.chunks.is_empty() {
        return Ok(Json(EmbedChunksResponse {
            chunks: vec![],
            model: state.config.model_name.clone(),
            provider: "local".to_string(),
        }));
    }

    // Extract texts from chunks
    let texts: Vec<String> = request.chunks.iter().map(|c| c.content.clone()).collect();
    let ids: Vec<String> = request.chunks.iter().map(|c| c.id.clone()).collect();

    match state
        .orchestrator
        .embed_batch(texts, request.source_type.as_deref(), None, None)
        .await
    {
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

            Ok(Json(EmbedChunksResponse {
                chunks: embedded_chunks,
                model: response.model,
                provider: "local".to_string(),
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
