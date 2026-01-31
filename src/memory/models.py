"""
Data models for the persistent memory system.
Phase 1: Core memory structures for daily logs, long-term memory, and semantic search.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class MemoryType(str, Enum):
    """Types of memories that can be stored"""
    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    EVENT = "event"
    CONVERSATION = "conversation"
    GENERAL = "general"


class MemoryConfidence(str, Enum):
    """Confidence levels for memory extraction"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MemoryEntry:
    """
    Core memory entry structure.
    Used for both daily logs and long-term memory storage.
    """
    content: str
    memory_type: MemoryType = MemoryType.GENERAL
    channel_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: MemoryConfidence = MemoryConfidence.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Semantic search fields
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['memory_type'] = self.memory_type.value if isinstance(self.memory_type, MemoryType) else self.memory_type
        data['confidence'] = self.confidence.value if isinstance(self.confidence, MemoryConfidence) else self.confidence
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryEntry':
        """Create from dictionary"""
        # Handle datetime conversion
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Handle enum conversions
        if 'memory_type' in data and isinstance(data['memory_type'], str):
            data['memory_type'] = MemoryType(data['memory_type'])
        if 'confidence' in data and isinstance(data['confidence'], str):
            data['confidence'] = MemoryConfidence(data['confidence'])
        
        return cls(**data)
    
    def to_markdown(self, include_metadata: bool = False) -> str:
        """
        Convert to markdown format for daily logs or MEMORY.md.
        
        Args:
            include_metadata: Whether to include metadata in output
        
        Returns:
            Markdown formatted string
        """
        timestamp_str = self.timestamp.strftime("%H:%M")
        type_str = self.memory_type.value.upper()
        
        lines = [f"**[{timestamp_str}] {type_str}**: {self.content}"]
        
        if include_metadata:
            if self.channel_id:
                lines.append(f"  - Channel: {self.channel_id}")
            if self.user_id:
                lines.append(f"  - User: {self.user_id}")
            if self.confidence != MemoryConfidence.MEDIUM:
                lines.append(f"  - Confidence: {self.confidence.value}")
            if self.metadata:
                for key, value in self.metadata.items():
                    lines.append(f"  - {key}: {value}")
        
        return "\n".join(lines)


@dataclass
class DailyLogEntry:
    """
    Entry for daily markdown logs.
    Wraps MemoryEntry with additional context for daily summaries.
    """
    memory: MemoryEntry
    channel_name: Optional[str] = None
    guild_name: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Convert to markdown format for daily log files"""
        header_parts = [self.memory.timestamp.strftime("%H:%M")]
        
        if self.channel_name:
            header_parts.append(f"Channel: {self.channel_name}")
        elif self.memory.channel_id:
            header_parts.append(f"Channel ID: {self.memory.channel_id}")
        
        header = f"## [{' | '.join(header_parts)}]"
        
        lines = [header, "", self.memory.content]
        
        if self.memory.metadata:
            lines.append("")
            lines.append("**Metadata:**")
            for key, value in self.memory.metadata.items():
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)


@dataclass
class SearchResult:
    """
    Result from semantic memory search.
    Includes the memory entry and its relevance score.
    """
    memory: MemoryEntry
    score: float
    distance: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "distance": self.distance
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SearchResult':
        """Create from dictionary"""
        return cls(
            memory=MemoryEntry.from_dict(data['memory']),
            score=data['score'],
            distance=data.get('distance')
        )


@dataclass
class UserProfile:
    """
    User-specific profile information learned over time.
    Stored in USER_PROFILES.md
    """
    user_id: str
    username: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    facts: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = {
            "user_id": self.user_id,
            "username": self.username,
            "preferences": self.preferences,
            "facts": self.facts,
            "last_updated": self.last_updated.isoformat()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Create from dictionary"""
        if isinstance(data.get('last_updated'), str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)
    
    def to_markdown(self) -> str:
        """Convert to markdown format for USER_PROFILES.md"""
        lines = [
            f"## User: {self.username or self.user_id}",
            f"*Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M')}*",
            ""
        ]
        
        if self.preferences:
            lines.append("### Preferences")
            for key, value in self.preferences.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        if self.facts:
            lines.append("### Facts")
            for fact in self.facts:
                lines.append(f"- {fact}")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class ChannelProfile:
    """
    Channel-specific context and information.
    Stored in CHANNEL_PROFILES.md
    """
    channel_id: str
    channel_name: Optional[str] = None
    guild_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "guild_name": self.guild_name,
            "context": self.context,
            "topics": self.topics,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChannelProfile':
        """Create from dictionary"""
        if isinstance(data.get('last_updated'), str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)
    
    def to_markdown(self) -> str:
        """Convert to markdown format for CHANNEL_PROFILES.md"""
        lines = [
            f"## Channel: {self.channel_name or self.channel_id}",
            f"*Guild: {self.guild_name or 'Unknown'}*",
            f"*Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M')}*",
            ""
        ]
        
        if self.topics:
            lines.append("### Topics")
            for topic in self.topics:
                lines.append(f"- {topic}")
            lines.append("")
        
        if self.context:
            lines.append("### Context")
            for key, value in self.context.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        return "\n".join(lines)
