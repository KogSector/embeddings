//! Embedding clients module.

pub mod openai;
pub mod cohere;
pub mod voyage;
pub mod jina;
pub mod huggingface;

pub use openai::OpenAIClient;
pub use cohere::CohereClient;
pub use voyage::VoyageClient;
pub use jina::JinaClient;
pub use huggingface::HuggingFaceClient;
