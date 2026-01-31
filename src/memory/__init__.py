"""
Persistent Memory System - Phase 1

Provides hybrid memory storage combining:
- Daily markdown logs
- Long-term curated memory
- Semantic search via vector embeddings

Usage:
    from memory import HybridMemoryStorage, MemoryEntry, MemoryType
    
    # Initialize storage (auto-detects available embedding provider)
    storage = HybridMemoryStorage(base_path="memory")
    
    # Save a daily note
    memory = MemoryEntry(
        content="User mentioned they live in Pittsburgh",
        memory_type=MemoryType.FACT,
        channel_id="123456",
        user_id="789012"
    )
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
]

__version__ = "1.0.0-phase1"
