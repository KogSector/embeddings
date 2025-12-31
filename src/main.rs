//! Embeddings Service - Main Entry Point
//!
//! A high-performance embedding microservice using local ONNX models.
//! No external API dependencies - runs entirely on-device.

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use embeddings::config::Config;
use embeddings::handlers::{self, AppState};
use embeddings::services::EmbeddingOrchestrator;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "embeddings=info,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    dotenvy::dotenv().ok();
    let config = Config::from_env();

    info!("🚀 Starting Embeddings Service v{}", env!("CARGO_PKG_VERSION"));
    info!("📦 Local Model: {} ({}D)", config.model_name, config.model_dimension);
    info!("🔧 Port: {}", config.port);

    // Validate model files exist
    match config.validate_model_files() {
        Ok(_) => info!("✅ Model files validated"),
        Err(e) => {
            tracing::error!("❌ Model files missing: {}", e);
            tracing::error!("Please download the model files:");
            tracing::error!("  - Model: {}", config.model_path);
            tracing::error!("  - Tokenizer: {}", config.tokenizer_path);
            tracing::error!("");
            tracing::error!("You can download all-MiniLM-L6-v2 from:");
            tracing::error!("  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2");
            return Err(anyhow::anyhow!(e));
        }
    }

    // Initialize the embedding orchestrator
    let orchestrator = match EmbeddingOrchestrator::new(&config).await {
        Ok(orch) => {
            info!("✅ Local embedding model initialized");
            info!("   Cache size: {} entries", config.cache_size);
            Arc::new(orch)
        }
        Err(e) => {
            tracing::error!("Failed to initialize embedding orchestrator: {}", e);
            return Err(e);
        }
    };

    let state = Arc::new(AppState {
        orchestrator,
        config: config.clone(),
    });

    // Build HTTP routes
    let app = Router::new()
        // Health check
        .route("/health", get(handlers::health_check))
        // Embedding endpoints
        .route("/embed", post(handlers::embed_single))
        .route("/batch/embed", post(handlers::embed_batch))
        .route("/batch/embed/chunks", post(handlers::embed_chunks))
        // State
        .with_state(state)
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!("✅ Embeddings Service listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
