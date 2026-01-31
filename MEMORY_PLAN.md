# Persistent Memory Plan for ChatGPT-Discord-Bot

## Current Architecture Assessment

The bot already has a solid foundation with:
- `EnhancedConversationManager` - per-channel conversation contexts
- `JSONMemoryStorage` - basic JSON file persistence  
- Abstract `MemoryStorageInterface` for pluggable storage backends
- Metadata tracking (channel info, activity, message counts)

**What's missing for true persistent memory:**
1. Semantic search over memories (not just raw JSON retrieval)
2. Structured memory types (facts, preferences, decisions vs raw conversation)
3. Daily logs + curated long-term memory split
4. Automatic memory capture from conversations
5. Memory recall/injection into context

---

## Recommended Memory Structure

### 1. File Organization

```
memory/
├── daily/
│   ├── 2026-01-31.md          # Raw conversation summaries per day
│   ├── 2026-01-30.md
│   └── ...
├── long-term/
│   ├── MEMORY.md              # Curated distilled memories (like OpenClaw)
│   ├── USER_PROFILES.md       # User-specific learnings
│   └── CHANNEL_PROFILES.md    # Channel-specific context
├── search/
│   └── embeddings.sqlite      # Vector index for semantic search
└── raw/
    └── conversations/         # Current JSON storage (keep as backup)
```

### 2. Memory Types to Track

| Type | Example | Storage |
|------|---------|---------|
| **Facts** | "User Joe lives in Pittsburgh" | Long-term MEMORY.md + embeddings |
| **Preferences** | "Joe prefers concise answers" | USER_PROFILES.md |
| **Decisions** | "Decided to use GPT-5 for images" | Long-term MEMORY.md |
| **Events** | "Game notification set for Steelers" | Daily log + searchable |
| **Raw Chat** | Full conversation history | JSON (existing) |

---

## Implementation Plan

### Phase 1: Enhanced Storage Layer

Create `src/memory/` module:

```python
# src/memory/storage.py
class HybridMemoryStorage:
    """
    Combines JSON for raw data + Markdown for human-readable memories
    + SQLite for vector search
    """
    
    def __init__(self, base_path: str = "memory"):
        self.base_path = Path(base_path)
        self.daily_path = self.base_path / "daily"
        self.long_term_path = self.base_path / "long-term"
        self.vector_store = LanceDBStore(self.base_path / "search")
        
    async def save_daily_note(self, channel_id: str, content: str):
        """Append to today's daily log"""
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self.daily_path / f"{today}.md"
        
        timestamp = datetime.now().strftime("%H:%M")
        entry = f"\n## [{timestamp}] Channel {channel_id}\n\n{content}\n"
        
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(entry)
    
    async def add_long_term_memory(self, memory: MemoryEntry):
        """Add to curated MEMORY.md and vector index"""
        # Update markdown
        await self._append_to_memory_md(memory)
        # Add to vector DB for search
        await self.vector_store.index(memory)
```

### Phase 2: Memory Capture System

Hook into existing conversation flow:

```python
# In src/aclient.py - modify handle_response()

async def handle_response(self, ...):
    # ... existing code ...
    
    response = ""
    async for token in async_generator:
        response += token
    
    # NEW: Capture memory after response completes
    await self._capture_memory_if_important(channel_id, user_message, response)
    
    return response

async def _capture_memory_if_important(self, channel_id: str, user_msg: str, assistant_msg: str):
    """Use LLM to judge if this exchange contains memorable info"""
    
    memory_prompt = f"""Analyze this conversation exchange. Should any of this be remembered long-term?
    
    User: {user_msg}
    Assistant: {assistant_msg}
    
    If the user shared:
    - Personal facts (location, job, interests)
    - Preferences (communication style, likes/dislikes)  
    - Important decisions or plans
    - Key context about ongoing projects
    
    Then extract and categorize. Otherwise respond "NONE".
    
    Format:
    TYPE: [fact|preference|decision|event|none]
    CONTENT: [what to remember]
    CONFIDENCE: [high|medium|low]
    """
    
    result = await self.client.responses.create(
        model="gpt-5",
        instructions="You are a memory extraction system. Be conservative - only capture truly important information.",
        input=memory_prompt,
        temperature=0.1,
        max_output_tokens=500,
        text={"verbosity": "low"}
    )
    
    # Parse and store if valid memory found
    memory = self._parse_memory_output(result.output)
    if memory and memory.confidence == "high":
        await self.memory_storage.add_long_term_memory(memory)
```

### Phase 3: Memory Recall (Search)

Add to `EnhancedConversationManager`:

```python
# In conversation_manager.py

class EnhancedConversationManager:
    # ... existing code ...
    
    async def get_context_with_memory(self, channel_id: str, current_message: str) -> str:
        """Get conversation history + relevant memories"""
        
        # Get raw history
        history = await self.get_history(channel_id)
        
        # Search for relevant memories
        relevant_memories = await self.storage.search_memories(
            query=current_message,
            channel_id=channel_id,
            top_k=5
        )
        
        # Build enhanced context
        context_parts = []
        
        if relevant_memories:
            context_parts.append("## Relevant Memories\n")
            for mem in relevant_memories:
                context_parts.append(f"- {mem.content} (from {mem.date})")
            context_parts.append("\n")
        
        # Add recent conversation
        context_parts.append("## Recent Conversation\n")
        for msg in history[-10:]:  # Last 10 messages
            context_parts.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context_parts)
```

### Phase 4: Integration Points

#### A. Modify `aclient.py` - `ask_stream_async()`

Replace history formatting with memory-augmented version:

```python
async def ask_stream_async(self, message=None, channel_id=None):
    # ... 
    
    # OLD: formatted_input = await self._format_history_for_responses(channel_id)
    # NEW: Get history + relevant memories
    formatted_input = await self.conversation_manager.get_context_with_memory(
        channel_id, message
    )
    
    # ... rest of method unchanged
```

#### B. Add `/remember` Slash Command

```python
@client.tree.command(name="remember", description="Add a fact to my long-term memory")
async def remember(interaction: discord.Interaction, fact: str, category: str = "general"):
    """Manual memory entry"""
    await interaction.response.defer(ephemeral=True)
    
    memory = MemoryEntry(
        type=category,
        content=fact,
        channel_id=str(interaction.channel_id),
        user_id=str(interaction.user.id),
        timestamp=datetime.now()
    )
    
    await client.memory_storage.add_long_term_memory(memory)
    await interaction.followup.send(f"✅ I'll remember: {fact}", ephemeral=True)

@client.tree.command(name="recall", description="Search my memories")
async def recall(interaction: discord.Interaction, query: str):
    """Search memories"""
    await interaction.response.defer(ephemeral=False)
    
    memories = await client.memory_storage.search_memories(
        query=query,
        channel_id=str(interaction.channel_id)
    )
    
    if not memories:
        await interaction.followup.send("I don't have any memories about that.")
        return
    
    embed = discord.Embed(title=f"🔍 Memories: {query}")
    for mem in memories[:5]:
        embed.add_field(
            name=f"{mem.type} ({mem.date.strftime('%Y-%m-%d')})",
            value=mem.content[:100] + "..." if len(mem.content) > 100 else mem.content,
            inline=False
        )
    
    await interaction.followup.send(embed=embed)
```

#### C. Daily Log Automation

Add to `EnhancedConversationManager`:

```python
async def add_message(self, channel_id: str, message: dict):
    """Enhanced to also write daily logs"""
    # ... existing code ...
    
    # NEW: Append to daily log
    if message.get("role") == "user":
        await self.storage.save_daily_note(
            channel_id=channel_id,
            content=f"User: {message.get('content', '')[:200]}..."
        )
    elif message.get("role") == "assistant":
        await self.storage.save_daily_note(
            channel_id=channel_id, 
            content=f"Bot: {message.get('content', '')[:200]}..."
        )
```

---

## Example Memory Flow

### User Interaction:
```
User: "I'm flying to Vegas on Feb 5th, can you remind me about my hotel?"
Bot: "Your hotel is Resorts World Las Vegas, check-in Feb 5 at 4 PM..."
```

### Auto-Capture:
```
TYPE: event
CONTENT: User flying to Vegas Feb 5-9. Hotel: Resorts World Las Vegas, check-in Feb 5 @ 4pm.
CONFIDENCE: high
```

### Storage:
1. **Daily log** (`memory/daily/2026-01-30.md`):
   ```markdown
   ## [20:30] Channel lobster-tank
   
   User: I'm flying to Vegas on Feb 5th...
   Bot: Your hotel is Resorts World...
   ```

2. **Long-term memory** (`memory/long-term/MEMORY.md`):
   ```markdown
   ## Travel Plans
   - Vegas trip Feb 5-9, 2026: Resorts World Las Vegas, Confirmation LAPS162R9Q
   ```

3. **Vector index**: Embedding of "Vegas trip Feb 5..." for semantic search

### Later Query:
```
User: "When do I leave for Vegas?"
Bot: (searches memory, finds "Vegas trip Feb 5-9") 
    "You're flying to Las Vegas on February 5th."
```

---

## Technical Dependencies

```txt
# requirements.txt additions
lancedb>=0.5.0           # Vector database
sentence-transformers>=2.2.0  # For embeddings (or use OpenAI embeddings)
aiofiles>=23.0           # Async file operations
```

Or use **OpenAI embeddings** to avoid local models:
```python
# Use OpenAI for embeddings (simpler, but API cost)
class OpenAIEmbeddingStore:
    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
```

---

## Migration Path

1. **Phase 1** (Week 1): Implement storage layer, start writing daily logs
2. **Phase 2** (Week 2): Add automatic memory extraction
3. **Phase 3** (Week 3): Add memory recall to context
4. **Phase 4** (Week 4): Add slash commands, UI

---

## Optional Enhancements

1. **Heartbeat system**: Add a periodic task (like OpenClaw's `HEARTBEAT.md`) to:
   - Review recent memories
   - Clean up low-confidence entries
   - Surface time-sensitive reminders

2. **Memory consolidation**: Nightly job to:
   - Summarize daily logs
   - Promote important daily items to long-term memory
   - Archive old daily files

3. **User-specific profiles**: Track per-user preferences across channels

4. **Web UI**: Simple Flask/FastAPI to browse/search memories

---

## OpenClaw Parity Checklist

| Feature | OpenClaw | This Plan |
|---------|----------|-----------|
| Daily notes (YYYY-MM-DD.md) | ✅ | ✅ |
| Long-term MEMORY.md | ✅ | ✅ |
| Semantic search | ✅ | ✅ (vector DB) |
| Auto-capture | ✅ | ✅ (LLM extraction) |
| Manual `/remember` | N/A | ✅ (Discord command) |
| Manual `/recall` | N/A | ✅ (Discord command) |
| Per-channel isolation | ✅ | ✅ |
| User profiles | Partial | ✅ |
| Heartbeat reminders | ✅ | Can add |

This plan gives the bot **memory parity with OpenClaw** plus Discord-native interfaces.
