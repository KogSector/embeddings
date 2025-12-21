# Build stage
FROM rust:1.75-slim-bookworm AS builder

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock* ./

# Create a dummy main.rs to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub mod config; pub mod traits; pub mod clients; pub mod services; pub mod handlers; pub mod models;" > src/lib.rs && \
    mkdir -p src/clients src/services && \
    touch src/config.rs src/traits.rs src/models.rs src/handlers.rs && \
    touch src/clients/mod.rs src/services/mod.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release 2>/dev/null || true

# Remove dummy files
RUN rm -rf src

# Copy actual source code
COPY src ./src
COPY config ./config

# Build the actual application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /app/target/release/embeddings /app/embeddings

# Copy configuration files
COPY --from=builder /app/config /app/config

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 3021

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3021/health || exit 1

# Run the service
ENV RUST_LOG=embeddings=info,tower_http=debug
CMD ["./embeddings"]
