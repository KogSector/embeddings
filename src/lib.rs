//! Embeddings Service - Library Entry Point
//!
//! A high-performance embedding microservice using local ONNX models.
//! No external API dependencies - runs entirely on-device.

pub mod clients;
pub mod config;
pub mod handlers;
pub mod models;
pub mod services;
pub mod traits;

// Re-export commonly used types
pub use clients::{LocalEmbeddingClient, LocalModelConfig};
pub use config::Config;
pub use services::EmbeddingOrchestrator;
pub use traits::{EmbeddingClient, EmbeddingRequest, EmbeddingResponse};
