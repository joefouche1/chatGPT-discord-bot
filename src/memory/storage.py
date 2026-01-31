"""
Hybrid memory storage implementation for Phase 1.

Combines:
- Daily markdown logs (memory/daily/YYYY-MM-DD.md)
- Long-term memory (memory/long-term/MEMORY.md)
- SQLite vector store (memory/search/embeddings.sqlite) for semantic search

Supports both OpenAI embeddings API and local sentence-transformers.
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod

import aiosqlite
import aiofiles

from .models import (
    MemoryEntry, 
    MemoryType, 
    MemoryConfidence,
    SearchResult,
    DailyLogEntry,
    UserProfile,
    ChannelProfile
)

# Optional imports for embeddings
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation"""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier"""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings using text-embedding-3-small"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        # Load API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536  # text-embedding-3-small dimension
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def get_dimension(self) -> int:
        return self._dimension
    
    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using sentence-transformers"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        return self._dimension
    
    def get_model_name(self) -> str:
        return f"sentence-transformers/{self.model_name}"


class VectorStore:
    """SQLite-based vector storage for semantic search"""
    
    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider):
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self._initialized = False
    
    async def initialize(self):
        """Create database schema if it doesn't exist"""
        if self._initialized:
            return
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    channel_id TEXT,
                    user_id TEXT,
                    timestamp TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    metadata TEXT,
                    embedding_model TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    memory_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            
            # Create indices for faster queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel_id ON memories(channel_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
            """)
            
            await db.commit()
        
        self._initialized = True
    
    async def add_memory(self, memory: MemoryEntry) -> int:
        """Add memory with embedding to database"""
        await self.initialize()
        
        # Generate embedding if not present
        if memory.embedding is None:
            memory.embedding = await self.embedding_provider.embed(memory.content)
            memory.embedding_model = self.embedding_provider.get_model_name()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO memories 
                (content, memory_type, channel_id, user_id, timestamp, confidence, metadata, embedding_model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.content,
                memory.memory_type.value if isinstance(memory.memory_type, MemoryType) else memory.memory_type,
                memory.channel_id,
                memory.user_id,
                memory.timestamp.isoformat(),
                memory.confidence.value if isinstance(memory.confidence, MemoryConfidence) else memory.confidence,
                json.dumps(memory.metadata) if memory.metadata else None,
                memory.embedding_model,
                datetime.now().isoformat()
            ))
            
            memory_id = cursor.lastrowid
            
            # Store embedding as blob
            embedding_bytes = json.dumps(memory.embedding).encode('utf-8')
            await db.execute("""
                INSERT INTO embeddings (memory_id, embedding, dimension)
                VALUES (?, ?, ?)
            """, (memory_id, embedding_bytes, len(memory.embedding)))
            
            await db.commit()
        
        return memory_id
    
    async def search_memories(
        self,
        query: str,
        top_k: int = 5,
        channel_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        min_confidence: Optional[MemoryConfidence] = None
    ) -> List[SearchResult]:
        """
        Search memories using cosine similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            channel_id: Filter by channel (optional)
            memory_type: Filter by memory type (optional)
            min_confidence: Minimum confidence level (optional)
        
        Returns:
            List of SearchResult objects sorted by relevance
        """
        await self.initialize()
        
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(query)
        
        # Build SQL query with filters
        where_clauses = []
        params = []
        
        if channel_id:
            where_clauses.append("m.channel_id = ?")
            params.append(channel_id)
        
        if memory_type:
            where_clauses.append("m.memory_type = ?")
            params.append(memory_type.value if isinstance(memory_type, MemoryType) else memory_type)
        
        if min_confidence:
            confidence_order = ["low", "medium", "high"]
            min_idx = confidence_order.index(
                min_confidence.value if isinstance(min_confidence, MemoryConfidence) else min_confidence
            )
            params.append(confidence_order[min_idx:])
            where_clauses.append(f"m.confidence IN ({','.join(['?'] * len(params[-1]))})")
            params = params[:-1] + params[-1]
        
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(f"""
                SELECT m.id, m.content, m.memory_type, m.channel_id, m.user_id, 
                       m.timestamp, m.confidence, m.metadata, m.embedding_model,
                       e.embedding
                FROM memories m
                JOIN embeddings e ON m.id = e.memory_id
                {where_clause}
            """, params)
            
            rows = await cursor.fetchall()
        
        # Calculate cosine similarity for each result
        results = []
        for row in rows:
            (mem_id, content, mem_type, chan_id, usr_id, ts, conf, meta, emb_model, emb_blob) = row
            
            # Deserialize embedding
            stored_embedding = json.loads(emb_blob.decode('utf-8'))
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            
            # Create MemoryEntry
            memory = MemoryEntry(
                content=content,
                memory_type=MemoryType(mem_type),
                channel_id=chan_id,
                user_id=usr_id,
                timestamp=datetime.fromisoformat(ts),
                confidence=MemoryConfidence(conf),
                metadata=json.loads(meta) if meta else {},
                embedding=stored_embedding,
                embedding_model=emb_model
            )
            
            results.append(SearchResult(
                memory=memory,
                score=similarity,
                distance=1.0 - similarity
            ))
        
        # Sort by score (descending) and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)


class HybridMemoryStorage:
    """
    Hybrid storage combining markdown files and vector search.
    
    Directory structure:
    memory/
    ├── daily/
    │   ├── 2026-01-31.md
    │   └── ...
    ├── long-term/
    │   ├── MEMORY.md
    │   ├── USER_PROFILES.md
    │   └── CHANNEL_PROFILES.md
    └── search/
        └── embeddings.sqlite
    """
    
    def __init__(
        self,
        base_path: Union[str, Path] = "memory",
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        self.base_path = Path(base_path)
        self.daily_path = self.base_path / "daily"
        self.long_term_path = self.base_path / "long-term"
        self.search_path = self.base_path / "search"
        
        # Create directories
        self.daily_path.mkdir(parents=True, exist_ok=True)
        self.long_term_path.mkdir(parents=True, exist_ok=True)
        self.search_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding provider (default to OpenAI if available)
        if embedding_provider is None:
            if OPENAI_AVAILABLE:
                embedding_provider = OpenAIEmbeddingProvider()
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                embedding_provider = SentenceTransformerEmbeddingProvider()
            else:
                raise RuntimeError(
                    "No embedding provider available. Install either 'openai' or 'sentence-transformers'"
                )
        
        self.embedding_provider = embedding_provider
        
        # Initialize vector store
        self.vector_store = VectorStore(
            self.search_path / "embeddings.sqlite",
            embedding_provider
        )
    
    async def save_daily_note(
        self,
        memory: MemoryEntry,
        channel_name: Optional[str] = None,
        guild_name: Optional[str] = None
    ):
        """
        Append to today's daily log.
        
        Args:
            memory: Memory entry to save
            channel_name: Optional channel name for context
            guild_name: Optional guild name for context
        """
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self.daily_path / f"{today}.md"
        
        # Create daily log entry
        log_entry = DailyLogEntry(
            memory=memory,
            channel_name=channel_name,
            guild_name=guild_name
        )
        
        # Append to file
        entry_text = "\n\n" + log_entry.to_markdown() + "\n"
        
        # Create file with header if it doesn't exist
        if not file_path.exists():
            header = f"# Daily Log - {today}\n\n"
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(header)
        
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(entry_text)
    
    async def add_long_term_memory(self, memory: MemoryEntry):
        """
        Add to curated MEMORY.md and vector index.
        
        Args:
            memory: Memory entry to store long-term
        """
        memory_file = self.long_term_path / "MEMORY.md"
        
        # Create file with header if it doesn't exist
        if not memory_file.exists():
            header = "# Long-Term Memory\n\n"
            async with aiofiles.open(memory_file, 'w') as f:
                await f.write(header)
        
        # Append memory
        timestamp = memory.timestamp.strftime("%Y-%m-%d")
        type_str = memory.memory_type.value.upper()
        entry_text = f"\n## [{timestamp}] {type_str}\n\n{memory.content}\n"
        
        if memory.metadata:
            entry_text += "\n**Context:**\n"
            for key, value in memory.metadata.items():
                entry_text += f"- {key}: {value}\n"
        
        async with aiofiles.open(memory_file, 'a') as f:
            await f.write(entry_text)
        
        # Add to vector index
        await self.vector_store.add_memory(memory)
    
    async def search_memories(
        self,
        query: str,
        top_k: int = 5,
        channel_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        min_confidence: Optional[MemoryConfidence] = None
    ) -> List[SearchResult]:
        """
        Search memories using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            channel_id: Filter by channel (optional)
            memory_type: Filter by memory type (optional)
            min_confidence: Minimum confidence level (optional)
        
        Returns:
            List of SearchResult objects sorted by relevance
        """
        return await self.vector_store.search_memories(
            query=query,
            top_k=top_k,
            channel_id=channel_id,
            memory_type=memory_type,
            min_confidence=min_confidence
        )
    
    async def get_recent_daily_logs(self, days: int = 7) -> Dict[str, str]:
        """
        Get recent daily logs.
        
        Args:
            days: Number of days to retrieve
        
        Returns:
            Dictionary mapping date to log content
        """
        logs = {}
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.daily_path / f"{date_str}.md"
            
            if file_path.exists():
                async with aiofiles.open(file_path, 'r') as f:
                    logs[date_str] = await f.read()
        
        return logs
    
    async def get_long_term_memory(self) -> str:
        """Get the contents of MEMORY.md"""
        memory_file = self.long_term_path / "MEMORY.md"
        
        if not memory_file.exists():
            return ""
        
        async with aiofiles.open(memory_file, 'r') as f:
            return await f.read()
    
    async def save_user_profile(self, profile: UserProfile):
        """Save or update user profile in USER_PROFILES.md"""
        profiles_file = self.long_term_path / "USER_PROFILES.md"
        
        # For now, append to file
        # TODO: Implement proper update logic
        if not profiles_file.exists():
            header = "# User Profiles\n\n"
            async with aiofiles.open(profiles_file, 'w') as f:
                await f.write(header)
        
        profile_text = "\n" + profile.to_markdown() + "\n---\n"
        
        async with aiofiles.open(profiles_file, 'a') as f:
            await f.write(profile_text)
    
    async def save_channel_profile(self, profile: ChannelProfile):
        """Save or update channel profile in CHANNEL_PROFILES.md"""
        profiles_file = self.long_term_path / "CHANNEL_PROFILES.md"
        
        if not profiles_file.exists():
            header = "# Channel Profiles\n\n"
            async with aiofiles.open(profiles_file, 'w') as f:
                await f.write(header)
        
        profile_text = "\n" + profile.to_markdown() + "\n---\n"
        
        async with aiofiles.open(profiles_file, 'a') as f:
            await f.write(profile_text)
    
    async def get_context_with_memory(
        self,
        channel_id: str,
        current_message: str,
        conversation_history: List[Dict[str, Any]],
        history_limit: int = 10
    ) -> str:
        """
        Phase 3: Get conversation context augmented with relevant memories.
        
        Args:
            channel_id: Channel ID to filter memories
            current_message: Current user message to search memories with
            conversation_history: Recent conversation history
            history_limit: Number of recent messages to include
        
        Returns:
            Formatted context string with memories + recent conversation
        """
        context_parts = []
        
        # Search for relevant memories using current message as query
        try:
            relevant_memories = await self.search_memories(
                query=current_message,
                top_k=5,
                channel_id=channel_id,
                min_confidence=MemoryConfidence.MEDIUM
            )
            
            if relevant_memories:
                context_parts.append("## Relevant Memories\n")
                for result in relevant_memories:
                    memory = result.memory
                    date_str = memory.timestamp.strftime("%Y-%m-%d")
                    # Format: - [memory content] (from [date])
                    context_parts.append(f"- {memory.content} (from {date_str})")
                context_parts.append("\n")
        except Exception as e:
            # Log error but continue with conversation history
            import logging
            logging.error(f"Error searching memories: {e}", exc_info=True)
        
        # Add recent conversation history
        context_parts.append("## Recent Conversation\n")
        
        # Take last N messages
        recent_messages = conversation_history[-history_limit:] if len(conversation_history) > history_limit else conversation_history
        
        for msg in recent_messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if isinstance(content, str):
                if role == "user":
                    context_parts.append(f"User: {content}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content}")
            elif isinstance(content, list):
                # Handle multimodal content
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                if text_parts:
                    combined_text = " ".join(text_parts)
                    if role == "user":
                        context_parts.append(f"User: {combined_text}")
                    elif role == "assistant":
                        context_parts.append(f"Assistant: {combined_text}")
        
        return "\n".join(context_parts)
