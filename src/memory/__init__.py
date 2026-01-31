"""
Persistent Memory System - Phase 1 & 2

Provides hybrid memory storage combining:
- Daily markdown logs
- Long-term curated memory
- Semantic search via vector embeddings
- Automatic memory extraction from conversations (Phase 2)

Usage:
    from memory import HybridMemoryStorage, MemoryEntry, MemoryType, MemoryExtractor
    
    # Initialize storage (auto-detects available embedding provider)
    storage = HybridMemoryStorage(base_path="memory")
    
    # Initialize extractor (Phase 2)
    extractor = MemoryExtractor(openai_client)
    
    # Extract memory from conversation
    memory = await extractor.extract_memory(
        user_message="I live in Pittsburgh",
        assistant_message="Great! Pittsburgh has a vibrant tech scene...",
        channel_id="123456",
        user_id="789012"
    )
    
    # Save a daily note
    if memory:
        await storage.save_daily_note(memory)
        # Add to long-term memory (also indexes for search)
        await storage.add_long_term_memory(memory)
    
    # Search memories
    results = await storage.search_memories("Where does the user live?", top_k=5)
"""

from .models import (
    MemoryEntry,
    MemoryType,
    MemoryConfidence,
    SearchResult,
    DailyLogEntry,
    UserProfile,
    ChannelProfile
)

from .storage import (
    HybridMemoryStorage,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    VectorStore
)

from .extractor import (
    MemoryExtractor,
    extract_and_store_memory
)

__all__ = [
    # Models
    "MemoryEntry",
    "MemoryType",
    "MemoryConfidence",
    "SearchResult",
    "DailyLogEntry",
    "UserProfile",
    "ChannelProfile",
    
    # Storage
    "HybridMemoryStorage",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "VectorStore",
    
    # Extractor (Phase 2)
    "MemoryExtractor",
    "extract_and_store_memory",
]

__version__ = "1.0.0-phase2"
