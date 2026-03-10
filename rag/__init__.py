"""RAG-based book analysis pipeline.

Modules:
    config       — Configuration loading
    chunker      — Text chunking with chapter detection
    embeddings   — Ollama embedding client
    store        — ChromaDB vector store
    ingest       — Book ingestion pipeline
    retrieval    — Chunk retrieval and query
    llm          — LLM generation client
    analysis     — Per-chapter and full-book analysis
    synthesis    — Cross-book synthesis
    critic       — Claim verification
    inspect_utils — Debugging and inspection
"""
