# Embeddings Microservice

A high-performance embedding microservice for RAG pipelines with multi-provider support and intelligent routing.

## Features

- **Multi-Provider Support**: OpenAI, Cohere, Voyage AI, Jina AI, HuggingFace
- **Intelligent Routing**: Route requests to optimal models based on content type
- **Multi-Model Fusion**: Combine embeddings from multiple models using weighted average, concatenation, or max pooling
- **Caching**: LFU-based caching for faster repeated queries
- **Batch Processing**: Efficient batch embedding with automatic cache optimization
- **Reranking**: Document reranking support via Jina AI

## Quick Start

1. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env and add at least one API key
```

2. **Run the service**:
```bash
cargo run
```

3. **Test the health endpoint**:
```bash
curl http://localhost:3021/health
```

## API Endpoints

### Health Check
```
GET /health
```

### Single Text Embedding
```
POST /embed
Content-Type: application/json

{
  "text": "Hello, world!",
  "source_type": "text",
  "provider": "openai",
  "model": "text-embedding-3-small"
}
```

### Batch Embedding
```
POST /batch/embed
Content-Type: application/json

{
  "texts": ["Hello", "World"],
  "source_type": "github"
}
```

### Chunk Embedding
```
POST /batch/embed/chunks
Content-Type: application/json

{
  "chunks": [
    {"id": "chunk-1", "content": "First chunk of text"},
    {"id": "chunk-2", "content": "Second chunk of text"}
  ],
  "source_type": "notion"
}
```

### Reranking
```
POST /rerank
Content-Type: application/json

{
  "query": "What is machine learning?",
  "documents": ["ML is a subset of AI", "Cooking recipes", "Neural networks"],
  "top_n": 2
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Service port | `3021` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `COHERE_API_KEY` | Cohere API key | - |
| `VOYAGE_API_KEY` | Voyage AI API key | - |
| `JINA_API_KEY` | Jina AI API key | - |
| `HUGGINGFACE_API_TOKEN` | HuggingFace token | - |
| `DEFAULT_PROVIDER` | Default embedding provider | `openai` |
| `EMBEDDING_CACHE_SIZE` | Max cached embeddings | `10000` |

### Routing Configuration

Edit `config/fusion_config.json` to customize model routing based on source type:

```json
{
  "routing": [
    {"source": "github", "models": ["voyage_code"], "weights": [1.0]},
    {"source": "notion", "models": ["openai_small"], "weights": [1.0]}
  ]
}
```

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | text-embedding-3-small, text-embedding-3-large |
| Cohere | embed-english-v3.0, embed-multilingual-v3.0 |
| Voyage | voyage-3, voyage-3-large, voyage-code-3 |
| Jina | jina-embeddings-v3 |
| HuggingFace | all-MiniLM-L6-v2, bge-large-en-v1.5 |

## Docker

```bash
docker build -t conhub-embeddings .
docker run -p 3021:3021 --env-file .env conhub-embeddings
```

## Integration with ConHub

This service integrates with the ConHub ecosystem:
- **data-connector**: Sends content for embedding after ingestion
- **chunker**: Sends chunked content for embedding
- **vector_rag**: Stores embeddings in Qdrant for retrieval

## License

MIT