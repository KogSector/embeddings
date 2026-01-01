# Embeddings Service Documentation

## Overview

The embeddings service generates **vector embeddings** for ConFuse's knowledge pipeline. It supports multiple embedding providers, intelligent model routing based on content type, and caching for efficiency.

## Role in ConFuse

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CHUNKER                                     │
│                    Segmented content chunks                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ chunks
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 EMBEDDINGS (This Service)                            │
│                         Port: 3005                                   │
│                                                                      │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│   │  Providers  │   │   Router    │   │    Cache    │              │
│   │             │   │             │   │             │              │
│   │ • OpenAI    │   │ • Source    │   │ • LFU evict │              │
│   │ • Cohere    │   │   type      │   │ • TTL       │              │
│   │ • Voyage    │   │   routing   │   │ • Warm-up   │              │
│   │ • Jina      │   │ • Fallback  │   │             │              │
│   │ • HuggingFc │   │             │   │             │              │
│   └─────────────┘   └─────────────┘   └─────────────┘              │
│                                                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ vectors
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       RELATION-GRAPH                                 │
│              Store in Zilliz for similarity search                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Supported Providers

| Provider | Models | Dimensions | Best For |
|----------|--------|------------|----------|
| OpenAI | text-embedding-3-small, 3-large | 1536, 3072 | General purpose |
| Cohere | embed-english-v3.0, multilingual | 1024 | Multilingual |
| Voyage | voyage-3, voyage-code-3 | 1024 | Code-specific |
| Jina | jina-embeddings-v3 | 1024 | Reranking support |
| HuggingFace | all-MiniLM-L6-v2 | 384 | Local/privacy |

## API Endpoints

### POST /embed

Generate embedding for single text.

**Request:**
```json
{
  "text": "def authenticate(user, password):",
  "source_type": "code",
  "provider": "voyage",
  "model": "voyage-code-3"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "dimensions": 1024,
  "model": "voyage-code-3",
  "cached": false,
  "processing_time_ms": 45
}
```

### POST /batch/embed

Batch embedding for multiple texts.

**Request:**
```json
{
  "texts": ["text 1", "text 2", "text 3"],
  "source_type": "document"
}
```

### POST /batch/embed/chunks

Batch embedding for chunks (preserves IDs).

**Request:**
```json
{
  "chunks": [
    {"id": "chunk-1", "content": "First chunk..."},
    {"id": "chunk-2", "content": "Second chunk..."}
  ],
  "source_type": "code"
}
```

**Response:**
```json
{
  "embeddings": [
    {"id": "chunk-1", "embedding": [...], "dimensions": 1024},
    {"id": "chunk-2", "embedding": [...], "dimensions": 1024}
  ],
  "model": "voyage-code-3",
  "cached_count": 1,
  "processing_time_ms": 120
}
```

### POST /rerank

Rerank documents by relevance to query.

**Request:**
```json
{
  "query": "authentication flow",
  "documents": [
    "JWT tokens are stateless",
    "Cooking recipes for dinner",
    "OAuth2 authorization code flow"
  ],
  "top_n": 2
}
```

**Response:**
```json
{
  "results": [
    {"index": 2, "score": 0.95, "text": "OAuth2 authorization code flow"},
    {"index": 0, "score": 0.72, "text": "JWT tokens are stateless"}
  ]
}
```

## Intelligent Routing

Route requests to optimal models based on content type:

```json
{
  "routing": [
    {
      "source_type": "code",
      "provider": "voyage",
      "model": "voyage-code-3",
      "priority": 1
    },
    {
      "source_type": "document",
      "provider": "openai",
      "model": "text-embedding-3-small",
      "priority": 1
    },
    {
      "source_type": "*",
      "provider": "openai",
      "model": "text-embedding-3-small",
      "priority": 99
    }
  ]
}
```

## Caching Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING CACHE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Key: hash(text + model)                                           │
│   Value: embedding vector                                           │
│                                                                      │
│   Eviction: LFU (Least Frequently Used)                             │
│   Max Size: 10,000 entries (configurable)                           │
│   TTL: 1 hour (configurable)                                        │
│                                                                      │
│   Cache Hit Rate: ~40% (typical)                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Benefits:
- Reduce API costs (cached embeddings are free)
- Lower latency for repeated content
- Automatic duplicate detection

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3005` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `COHERE_API_KEY` | Cohere API key | - |
| `VOYAGE_API_KEY` | Voyage API key | - |
| `JINA_API_KEY` | Jina API key | - |
| `HUGGINGFACE_API_TOKEN` | HuggingFace token | - |
| `DEFAULT_PROVIDER` | Default provider | `openai` |
| `EMBEDDING_CACHE_SIZE` | Max cached embeddings | `10000` |
| `CACHE_TTL_SECS` | Cache TTL | `3600` |

### Routing Configuration

Edit `config/fusion_config.json`:

```json
{
  "routing": [
    {"source": "github", "models": ["voyage_code"], "weights": [1.0]},
    {"source": "notion", "models": ["openai_small"], "weights": [1.0]},
    {"source": "slack", "models": ["cohere_english"], "weights": [1.0]}
  ],
  "fallback": "openai_small"
}
```

## Multi-Model Fusion

Combine embeddings from multiple models:

```json
{
  "fusion": {
    "strategy": "weighted_average",
    "models": [
      {"name": "openai_small", "weight": 0.6},
      {"name": "voyage_code", "weight": 0.4}
    ]
  }
}
```

Strategies:
- `weighted_average`: Weighted sum of normalized vectors
- `concatenation`: Concatenate vectors (higher dimensions)
- `max_pooling`: Element-wise max across models

## Error Handling

```json
{
  "error": {
    "code": "PROVIDER_ERROR",
    "message": "OpenAI API rate limit exceeded",
    "provider": "openai",
    "retryable": true,
    "retryAfter": 60
  }
}
```

Automatic fallback to secondary provider on failure.

## Performance

| Metric | Value |
|--------|-------|
| Throughput | ~100 embeddings/sec |
| Latency (cached) | <5ms |
| Latency (API) | 50-200ms |
| Cache hit rate | ~40% |

## Related Services

| Service | Relationship |
|---------|--------------|
| chunker | Sends chunks for embedding |
| relation-graph | Stores vectors in Zilliz |

## Cost Optimization

1. **Caching**: Avoid re-embedding identical content
2. **Batching**: Batch requests to reduce per-call overhead
3. **Model selection**: Use smaller models for less critical content
4. **Local models**: Use HuggingFace for privacy-sensitive data
