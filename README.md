# ConFuse Embeddings

Vector embedding service for the ConFuse Knowledge Intelligence Platform. Generates high-quality embeddings with multi-provider support and intelligent routing.

## Overview

This service generates **vector embeddings** that:
- Power semantic search across all knowledge
- Support multiple providers (OpenAI, Cohere, Voyage, Jina)
- Intelligently route based on content type
- Cache embeddings for efficiency

## Architecture

See [docs/README.md](docs/README.md) for complete documentation.

## Quick Start

```bash
# Build
cargo build --release

# Configure
cp .env.example .env
# Add at least one API key

# Run
cargo run
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Single text embedding |
| `/batch/embed` | POST | Batch embedding |
| `/batch/embed/chunks` | POST | Embed chunks |
| `/rerank` | POST | Rerank documents |
| `/health` | GET | Health check |

## Supported Providers

| Provider | Best For |
|----------|----------|
| OpenAI | General purpose |
| Cohere | Multilingual |
| Voyage | Code |
| Jina | Reranking |
| HuggingFace | Local/privacy |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI key |
| `VOYAGE_API_KEY` | Voyage key |
| `DEFAULT_PROVIDER` | Default provider |
| `EMBEDDING_CACHE_SIZE` | Cache size |

## Documentation

See [docs/](docs/) folder for complete documentation.

## License

MIT - ConFuse Team