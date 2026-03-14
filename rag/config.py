"""Configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EmbeddingConfig:
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"


@dataclass
class GenerationConfig:
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"


@dataclass
class ChunkingConfig:
    target_size: int = 1500
    overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000


@dataclass
class SummarizationConfig:
    window_size: int = 8000
    window_overlap: int = 500
    max_windows_per_section: int = 15
    quality: str = "default"        # "fast", "default", "thorough"
    verify_by_default: bool = False
    cache_dir: str = "./data/cache"


@dataclass
class RetrievalConfig:
    top_k: int = 10
    similarity_threshold: float = 0.0
    max_context_chunks: int = 20


@dataclass
class VectorStoreConfig:
    type: str = "chroma"
    persist_directory: str = "./data/vectorstore"
    collection_name: str = "books"


@dataclass
class StorageConfig:
    results_directory: str = "./data/results"


@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def load_config(path: str = "rag_config.yaml") -> RAGConfig:
    """Load configuration from YAML, falling back to defaults for missing keys."""
    config_path = Path(path)
    if not config_path.exists():
        return RAGConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    config = RAGConfig()
    section_map = {
        "embedding": EmbeddingConfig,
        "generation": GenerationConfig,
        "chunking": ChunkingConfig,
        "summarization": SummarizationConfig,
        "retrieval": RetrievalConfig,
        "vectorstore": VectorStoreConfig,
        "storage": StorageConfig,
    }
    for name, cls in section_map.items():
        if name in raw and isinstance(raw[name], dict):
            known_fields = {k for k in cls.__dataclass_fields__}
            filtered = {k: v for k, v in raw[name].items() if k in known_fields}
            setattr(config, name, cls(**filtered))

    return config
