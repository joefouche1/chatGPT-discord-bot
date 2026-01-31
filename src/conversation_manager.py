"""
Conversation Manager for handling per-channel conversation contexts.
Designed with future persistent memory storage in mind.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import os
from utils.log import logger


@dataclass
class ConversationMetadata:
    """Metadata for tracking conversation statistics and state"""
    channel_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    total_tokens_used: int = 0
    channel_name: Optional[str] = None
    guild_name: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        return cls(**data)


class MemoryStorageInterface(ABC):
    """
    Abstract interface for persistent memory storage.
    Future implementations can include: JSON file, SQLite, Redis, PostgreSQL, etc.
    """
    
    @abstractmethod
    async def save_context(self, channel_id: str, context_data: dict):
        """Save conversation context to persistent storage"""
        pass
    
    @abstractmethod
    async def load_context(self, channel_id: str) -> Optional[dict]:
        """Load conversation context from persistent storage"""
        pass
    
    @abstractmethod
    async def delete_context(self, channel_id: str):
        """Delete conversation context from persistent storage"""
        pass
    
    @abstractmethod
    async def list_contexts(self) -> List[str]:
        """List all stored context channel IDs"""
        pass
    
    @abstractmethod
    async def save_memory(self, channel_id: str, memory_type: str, memory_data: Any):
        """Save a specific memory/learning from conversation"""
        pass
    
    @abstractmethod
    async def get_memories(self, channel_id: str, memory_type: Optional[str] = None) -> List[dict]:
        """Retrieve memories for a channel, optionally filtered by type"""
        pass


class JSONMemoryStorage(MemoryStorageInterface):
    """Simple JSON file-based storage implementation"""
    
    def __init__(self, storage_dir: str = "conversation_memories"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.memories_dir = os.path.join(storage_dir, "memories")
        os.makedirs(self.memories_dir, exist_ok=True)
    
    def _get_context_path(self, channel_id: str) -> str:
        return os.path.join(self.storage_dir, f"context_{channel_id}.json")
    
    def _get_memories_path(self, channel_id: str) -> str:
        return os.path.join(self.memories_dir, f"memories_{channel_id}.json")
    
    async def save_context(self, channel_id: str, context_data: dict):
        """Save conversation context to JSON file"""
        try:
            path = self._get_context_path(channel_id)
            with open(path, 'w') as f:
                json.dump(context_data, f, indent=2, default=str)
            logger.info(f"Saved context for channel {channel_id}")
        except Exception as e:
            logger.error(f"Error saving context for channel {channel_id}: {e}")
    
    async def load_context(self, channel_id: str) -> Optional[dict]:
        """Load conversation context from JSON file"""
        try:
            path = self._get_context_path(channel_id)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading context for channel {channel_id}: {e}")
        return None
    
    async def delete_context(self, channel_id: str):
        """Delete conversation context file"""
        try:
            path = self._get_context_path(channel_id)
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted context for channel {channel_id}")
        except Exception as e:
            logger.error(f"Error deleting context for channel {channel_id}: {e}")
    
    async def list_contexts(self) -> List[str]:
        """List all stored context channel IDs"""
        contexts = []
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.startswith("context_") and filename.endswith(".json"):
                    channel_id = filename[8:-5]  # Remove "context_" and ".json"
                    contexts.append(channel_id)
        except Exception as e:
            logger.error(f"Error listing contexts: {e}")
        return contexts
    
    async def save_memory(self, channel_id: str, memory_type: str, memory_data: Any):
        """Save a specific memory/learning from conversation"""
        try:
            path = self._get_memories_path(channel_id)
            memories = []
            
            if os.path.exists(path):
                with open(path, 'r') as f:
                    memories = json.load(f)
            
            memory_entry = {
                "type": memory_type,
                "data": memory_data,
                "timestamp": datetime.now().isoformat()
            }
            memories.append(memory_entry)
            
            with open(path, 'w') as f:
                json.dump(memories, f, indent=2, default=str)
            
            logger.info(f"Saved {memory_type} memory for channel {channel_id}")
        except Exception as e:
            logger.error(f"Error saving memory for channel {channel_id}: {e}")
    
    async def get_memories(self, channel_id: str, memory_type: Optional[str] = None) -> List[dict]:
        """Retrieve memories for a channel, optionally filtered by type"""
        try:
            path = self._get_memories_path(channel_id)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    memories = json.load(f)
                    
                if memory_type:
                    return [m for m in memories if m.get("type") == memory_type]
                return memories
        except Exception as e:
            logger.error(f"Error loading memories for channel {channel_id}: {e}")
        return []


class ChannelConversationContext:
    """Individual conversation context for a specific channel"""
    
    def __init__(self, channel_id: str, channel_name: Optional[str] = None, 
                 guild_name: Optional[str] = None):
        self.channel_id = channel_id
        self.conversation_history: List[dict] = []
        self.metadata = ConversationMetadata(
            channel_id=channel_id,
            channel_name=channel_name,
            guild_name=guild_name
        )
        self.lock = asyncio.Lock()
        
        # Track important learned information for potential persistence
        self.learned_preferences: Dict[str, Any] = {}
        self.important_facts: List[str] = []
        self.user_profiles: Dict[str, dict] = {}  # User ID -> profile info
    
    def add_message(self, message: dict):
        """Add a message to the conversation history"""
        self.conversation_history.append(message)
        self.metadata.message_count += 1
        self.metadata.last_activity = datetime.now()
    
    def get_history(self) -> List[dict]:
        """Get the full conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history while preserving metadata"""
        self.conversation_history = []
        logger.info(f"Cleared history for channel {self.channel_id}")
    
    def truncate_history(self, keep_messages: int = 10):
        """Keep only the most recent messages"""
        if len(self.conversation_history) > keep_messages:
            self.conversation_history = self.conversation_history[-keep_messages:]
            logger.info(f"Truncated history for channel {self.channel_id} to {keep_messages} messages")
    
    def add_learned_preference(self, key: str, value: Any):
        """Store a learned preference about this channel's users"""
        self.learned_preferences[key] = value
        self.metadata.last_activity = datetime.now()
    
    def add_important_fact(self, fact: str):
        """Store an important fact learned in this conversation"""
        self.important_facts.append(fact)
        self.metadata.last_activity = datetime.now()
    
    def update_user_profile(self, user_id: str, profile_data: dict):
        """Update stored information about a specific user"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        self.user_profiles[user_id].update(profile_data)
        self.metadata.last_activity = datetime.now()
    
    def to_dict(self) -> dict:
        """Serialize context to dictionary for storage"""
        return {
            "channel_id": self.channel_id,
            "conversation_history": self.conversation_history,
            "metadata": self.metadata.to_dict(),
            "learned_preferences": self.learned_preferences,
            "important_facts": self.important_facts,
            "user_profiles": self.user_profiles
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChannelConversationContext':
        """Deserialize context from dictionary"""
        context = cls(data["channel_id"])
        context.conversation_history = data.get("conversation_history", [])
        context.metadata = ConversationMetadata.from_dict(data["metadata"])
        context.learned_preferences = data.get("learned_preferences", {})
        context.important_facts = data.get("important_facts", [])
        context.user_profiles = data.get("user_profiles", {})
        return context


class EnhancedConversationManager:
    """
    Enhanced conversation manager with per-channel contexts and 
    preparation for persistent memory storage.
    """
    
    def __init__(self, storage: Optional[MemoryStorageInterface] = None,
                 enable_persistence: bool = False,
                 max_inactive_hours: int = 168):  # 7 days
        self.contexts: Dict[str, ChannelConversationContext] = {}
        self.storage = storage or JSONMemoryStorage()
        self.enable_persistence = enable_persistence
        self.max_inactive_hours = max_inactive_hours
        self.global_lock = asyncio.Lock()
        
        # Global memories that apply across all channels
        self.global_memories: Dict[str, Any] = {}
        
    async def get_or_create_context(self, channel_id: str, 
                                   channel_name: Optional[str] = None,
                                   guild_name: Optional[str] = None) -> ChannelConversationContext:
        """Get existing context or create new one for a channel"""
        async with self.global_lock:
            if channel_id not in self.contexts:
                # Try to load from storage if persistence is enabled
                if self.enable_persistence:
                    stored_data = await self.storage.load_context(channel_id)
                    if stored_data:
                        self.contexts[channel_id] = ChannelConversationContext.from_dict(stored_data)
                        logger.info(f"Loaded persisted context for channel {channel_id}")
                        return self.contexts[channel_id]
                
                # Create new context
                self.contexts[channel_id] = ChannelConversationContext(
                    channel_id, channel_name, guild_name
                )
                logger.info(f"Created new context for channel {channel_id} ({channel_name})")
            
            return self.contexts[channel_id]
    
    async def add_message(self, channel_id: str, message: dict):
        """Add a message to a specific channel's context"""
        context = await self.get_or_create_context(channel_id)
        async with context.lock:
            context.add_message(message)
            
            # Auto-save if persistence is enabled
            if self.enable_persistence:
                await self.storage.save_context(channel_id, context.to_dict())
    
    async def get_history(self, channel_id: str) -> List[dict]:
        """Get conversation history for a specific channel"""
        context = await self.get_or_create_context(channel_id)
        return context.get_history()
    
    async def clear_context(self, channel_id: str):
        """Clear a specific channel's conversation context"""
        if channel_id in self.contexts:
            async with self.contexts[channel_id].lock:
                self.contexts[channel_id].clear_history()
                
                if self.enable_persistence:
                    await self.storage.delete_context(channel_id)
            
            logger.info(f"Cleared context for channel {channel_id}")
    
    async def save_memory(self, channel_id: str, memory_type: str, memory_data: Any):
        """Save an important memory/learning from conversation"""
        if self.enable_persistence:
            await self.storage.save_memory(channel_id, memory_type, memory_data)
    
    async def get_channel_stats(self, channel_id: str) -> Optional[dict]:
        """Get statistics for a specific channel's conversation"""
        if channel_id in self.contexts:
            context = self.contexts[channel_id]
            return {
                "channel_id": channel_id,
                "channel_name": context.metadata.channel_name,
                "guild_name": context.metadata.guild_name,
                "message_count": context.metadata.message_count,
                "created_at": context.metadata.created_at,
                "last_activity": context.metadata.last_activity,
                "history_length": len(context.conversation_history),
                "learned_preferences": len(context.learned_preferences),
                "important_facts": len(context.important_facts),
                "known_users": len(context.user_profiles)
            }
        return None
    
    async def cleanup_inactive_contexts(self):
        """Remove contexts that have been inactive for too long"""
        current_time = datetime.now()
        channels_to_remove = []
        
        async with self.global_lock:
            for channel_id, context in self.contexts.items():
                time_inactive = current_time - context.metadata.last_activity
                hours_inactive = time_inactive.total_seconds() / 3600
                
                if hours_inactive > self.max_inactive_hours:
                    channels_to_remove.append(channel_id)
            
            for channel_id in channels_to_remove:
                # Save important memories before removing
                if self.enable_persistence and self.contexts[channel_id].important_facts:
                    await self.storage.save_memory(
                        channel_id, 
                        "cleanup_summary",
                        {
                            "facts": self.contexts[channel_id].important_facts,
                            "preferences": self.contexts[channel_id].learned_preferences,
                            "cleanup_time": current_time.isoformat()
                        }
                    )
                
                del self.contexts[channel_id]
                logger.info(f"Removed inactive context for channel {channel_id}")
        
        return len(channels_to_remove)
    
    def get_all_contexts_info(self) -> List[dict]:
        """Get information about all active contexts"""
        info = []
        for channel_id, context in self.contexts.items():
            info.append({
                "channel_id": channel_id,
                "channel_name": context.metadata.channel_name,
                "message_count": context.metadata.message_count,
                "last_activity": context.metadata.last_activity.isoformat()
            })
        return info
    
    async def get_context_with_memory(
        self,
        channel_id: str,
        current_message: str,
        memory_storage=None,
        history_limit: int = 10
    ) -> str:
        """
        Phase 3: Get conversation history augmented with relevant memories.
        
        Args:
            channel_id: Channel ID
            current_message: Current user message to search memories with
            memory_storage: HybridMemoryStorage instance (optional)
            history_limit: Number of recent messages to include
        
        Returns:
            Formatted context string with memories + recent conversation
        """
        # Get conversation history
        conversation_history = await self.get_history(channel_id)
        
        # If memory storage is available and enabled, use it to augment context
        if memory_storage and hasattr(memory_storage, 'get_context_with_memory'):
            return await memory_storage.get_context_with_memory(
                channel_id=channel_id,
                current_message=current_message,
                conversation_history=conversation_history,
                history_limit=history_limit
            )
        
        # Fallback: return conversation history only
        context_parts = ["## Recent Conversation\n"]
        
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
