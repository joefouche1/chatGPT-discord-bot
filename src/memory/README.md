# Persistent Memory System - Phases 1 & 2

This module implements a hybrid memory storage system for the Discord bot, combining daily markdown logs, long-term curated memory, semantic search capabilities, and **automatic memory extraction from conversations** (Phase 2).

## Architecture

The system uses three storage mechanisms:

1. **Daily Logs** (`memory/daily/YYYY-MM-DD.md`)
   - Raw conversation summaries organized by date
   - Timestamped entries with channel context
   - Easy to review and audit

2. **Long-term Memory** (`memory/long-term/MEMORY.md`)
   - Curated, important memories
   - Categorized by type (facts, preferences, decisions, events)
   - Human-readable markdown format

3. **Vector Search** (`memory/search/embeddings.sqlite`)
   - SQLite database with semantic embeddings
   - Enables semantic similarity search
   - Fast retrieval of relevant memories

## Installation

Required dependencies:
```bash
pip install aiosqlite aiofiles openai
```

Or for local embeddings:
```bash
pip install aiosqlite aiofiles sentence-transformers
```

## Usage

### Basic Example

```python
from memory import HybridMemoryStorage, MemoryEntry, MemoryType

# Initialize storage
storage = HybridMemoryStorage(base_path="memory")

# Create a memory entry
memory = MemoryEntry(
    content="User prefers concise technical explanations",
    memory_type=MemoryType.PREFERENCE,
    channel_id="123456",
    user_id="789012"
)

# Save to daily log
await storage.save_daily_note(memory, channel_name="general")

# Add to long-term memory (also indexes for search)
await storage.add_long_term_memory(memory)

# Search memories
results = await storage.search_memories(
    query="What are the user's preferences?",
    top_k=5
)

for result in results:
    print(f"{result.memory.content} (score: {result.score})")
```

### Memory Types

```python
from memory import MemoryType

MemoryType.FACT          # Factual information
MemoryType.PREFERENCE    # User preferences
MemoryType.DECISION      # Decisions made
MemoryType.EVENT         # Events or activities
MemoryType.CONVERSATION  # Conversation snippets
MemoryType.GENERAL       # General memories
```

### Filtered Search

```python
# Search only preferences
results = await storage.search_memories(
    query="user likes",
    memory_type=MemoryType.PREFERENCE,
    top_k=5
)

# Search in specific channel
results = await storage.search_memories(
    query="recent discussion",
    channel_id="123456",
    top_k=5
)

# Minimum confidence filter
from memory import MemoryConfidence

results = await storage.search_memories(
    query="important facts",
    min_confidence=MemoryConfidence.HIGH,
    top_k=5
)
```

### Embedding Providers

The system supports two embedding providers:

#### OpenAI (default)
```python
from memory import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(
    api_key="sk-...",  # Optional, reads from env
    model="text-embedding-3-small"
)

storage = HybridMemoryStorage(
    base_path="memory",
    embedding_provider=provider
)
```

#### Sentence Transformers (local)
```python
from memory import SentenceTransformerEmbeddingProvider

provider = SentenceTransformerEmbeddingProvider(
    model_name="all-MiniLM-L6-v2"
)

storage = HybridMemoryStorage(
    base_path="memory",
    embedding_provider=provider
)
```

## File Structure

```
memory/
├── daily/
│   ├── 2026-01-31.md
│   ├── 2026-01-30.md
│   └── ...
├── long-term/
│   ├── MEMORY.md
│   ├── USER_PROFILES.md (future)
│   └── CHANNEL_PROFILES.md (future)
└── search/
    └── embeddings.sqlite
```

## Models

### MemoryEntry

Core data structure for memories:

```python
@dataclass
class MemoryEntry:
    content: str                          # The actual memory content
    memory_type: MemoryType               # Type of memory
    channel_id: Optional[str] = None      # Discord channel ID
    user_id: Optional[str] = None         # Discord user ID
    timestamp: datetime                   # When memory was created
    confidence: MemoryConfidence          # Confidence level
    metadata: Dict[str, Any]              # Additional context
    embedding: Optional[List[float]]      # Vector embedding
    embedding_model: Optional[str]        # Model used for embedding
```

### SearchResult

Result from semantic search:

```python
@dataclass
class SearchResult:
    memory: MemoryEntry    # The memory entry
    score: float           # Similarity score (0-1)
    distance: Optional[float]  # Distance metric
```

## API Reference

### HybridMemoryStorage

Main storage class:

- `save_daily_note(memory, channel_name, guild_name)` - Append to daily log
- `add_long_term_memory(memory)` - Add to MEMORY.md and vector index
- `search_memories(query, top_k, channel_id, memory_type, min_confidence)` - Semantic search
- `get_recent_daily_logs(days)` - Retrieve recent daily logs
- `get_long_term_memory()` - Get MEMORY.md contents
- `save_user_profile(profile)` - Save user profile
- `save_channel_profile(profile)` - Save channel profile

## Phase 2: Automatic Memory Extraction

Phase 2 adds automatic extraction of memorable information from conversations using LLM analysis.

### Usage

```python
from memory import MemoryExtractor, extract_and_store_memory

# Initialize extractor with OpenAI client
extractor = MemoryExtractor(openai_client)

# Extract memory from conversation
memory = await extractor.extract_memory(
    user_message="I live in Pittsburgh and work as a software engineer",
    assistant_message="Great! Pittsburgh has a vibrant tech scene...",
    channel_id="123456",
    user_id="789012",
    username="john_doe"
)

# If memory is HIGH confidence, it will be returned
if memory:
    print(f"Extracted {memory.memory_type}: {memory.content}")
```

### Helper Function

The `extract_and_store_memory` function combines extraction and storage:

```python
# Extract and store in one call
await extract_and_store_memory(
    extractor=memory_extractor,
    storage=memory_storage,
    user_message="I'm flying to Vegas on Feb 5th",
    assistant_message="Have a great trip!",
    channel_id="123456",
    user_id="789012",
    username="john_doe",
    channel_name="travel-plans",
    guild_name="My Server"
)
```

### Integration with Discord Bot

The bot automatically captures memories after each response:

```python
# In aclient.py handle_response():
if self.memory_extraction_enabled and user_message and response:
    asyncio.create_task(
        extract_and_store_memory(
            extractor=self.memory_extractor,
            storage=self.memory_storage,
            user_message=user_message,
            assistant_message=response,
            channel_id=channel_id,
            user_id=user_id,
            username=username,
            channel_name=channel_name,
            guild_name=guild_name
        )
    )
```

### Environment Variables

Control memory extraction:

```bash
# Enable/disable memory extraction (default: true)
ENABLE_MEMORY_EXTRACTION=true
```

### What Gets Extracted

The extractor identifies and captures:

- **Facts**: Personal information (location, job, interests, technical details)
- **Preferences**: Communication style, likes/dislikes, response preferences
- **Decisions**: Important choices, plans, commitments
- **Events**: Scheduled activities, deadlines, travel plans

**Only HIGH confidence extractions** are saved to avoid noise.

### Example Extractions

```
User: "I live in Pittsburgh and work as a software engineer"
→ TYPE: fact
→ CONTENT: User lives in Pittsburgh, works as software engineer
→ CONFIDENCE: high

User: "I prefer concise answers without too much explanation"
→ TYPE: preference
→ CONTENT: Prefers concise, brief responses without extensive explanations
→ CONFIDENCE: high

User: "I'm flying to Vegas on Feb 5th for a conference"
→ TYPE: event
→ CONTENT: Flying to Vegas on Feb 5th for a conference
→ CONFIDENCE: high
```

### How It Works

1. After each bot response, the conversation is analyzed
2. LLM extracts memorable information using structured prompt
3. Response is parsed into TYPE, CONTENT, CONFIDENCE
4. Only HIGH confidence memories are saved
5. Memory is added to daily log and long-term storage
6. Vector embedding is created for semantic search

## Next Phases

Phase 1 & 2 provide the foundation and automatic capture. Future phases will add:

- **Phase 3**: Memory recall and context injection into conversations
- **Phase 4**: Discord slash commands (`/remember`, `/recall`)
- **Phase 5**: Periodic consolidation and cleanup

## Testing

Run the test script to verify functionality:

```bash
python test_memory_phase1.py
```

This will:
1. Initialize the storage system
2. Create sample memories
3. Test daily logging
4. Test long-term memory
5. Test semantic search
6. Verify all files are created correctly

## License

Part of the chatGPT-discord-bot project.
