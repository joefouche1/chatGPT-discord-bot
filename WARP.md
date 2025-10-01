# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a multi-featured Discord bot powered by OpenAI's GPT models, originally based on ChatGPT integration but evolved to support multiple AI models (ChatGPT, Google Bard, Microsoft Bing) and advanced conversation management. The bot supports per-channel conversation contexts, slash commands, image generation, weather/news/sports data, and a customizable personality system.

## Architecture

### Core Components

**Main Entry Point**: `main.py` → `src/bot.py` → `src/aclient.py`

**Enhanced Conversation Management**: The bot uses a sophisticated per-channel conversation system:
- `src/conversation_manager.py` - Contains `EnhancedConversationManager` and `ChannelConversationContext` for isolated conversations per Discord channel
- `JSONMemoryStorage` class provides file-based persistence (extensible interface for future database backends)
- Automatic context cleanup for inactive channels
- Optional persistent storage via `ENABLE_PERSISTENCE` environment variable

**Bot Client**: `src/aclient.py` 
- Extends `discord.commands.Bot` 
- Manages OpenAI API interactions via `AsyncOpenAI`
- Handles message queuing and rate limiting
- Integrates conversation manager for per-channel context
- Image caching system for Discord attachments

**Command System**: Located in `src/commands/` directory
- `actions.py` - Action code processor for dynamic commands (!WEATHER, !SPORTS, etc.)
- `weather.py`, `news.py`, `sports.py` - External API integrations  
- `meme.py`, `context.py`, `tictactoe.py` - Interactive features
- Commands are loaded as Discord cogs

**Personality System**: `system_prompt.txt` defines "Megapig" character
- Edgy, humorous personality with domain expertise
- Action codes for real-time data (!WEATHER, !SPORTS, !NEWS, !DRAW)
- Contextual responses with Trek references and culinary advice

### Key Architectural Decisions

**Per-Channel Isolation**: Each Discord channel maintains its own conversation history, learned preferences, and user profiles. This prevents context bleeding between different channels/servers.

**Extensible Storage**: The `MemoryStorageInterface` abstract class allows easy implementation of different storage backends (currently JSON files, designed for future database integration).

**Async Processing**: Message queue system prevents blocking on long API calls, with dedicated message processing loop.

**Modular Commands**: Slash commands and cogs system allows easy addition of new features without core bot modifications.

## Development Commands

### Environment Setup
```bash
# Create virtual environment (Python 3.13+ required)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
# or for development with optional dependencies
pip install -e .[dev]
```

### Running the Bot
```bash
# Local development
python main.py

# With Docker
docker compose up -d
docker logs -f chatgpt-discord-bot

# Stop Docker
docker compose down
```

### Development Tools
```bash
# Code formatting
black src/ utils/ main.py

# Linting 
flake8 src/ utils/ main.py

# Type checking
mypy src/ utils/ main.py

# Testing
pytest
pytest tests/unit/
pytest -m "not slow"  # Skip slow tests
```

### Dependency Management
```bash
# Update dependencies (uses uv)
uv pip compile requirements.in --output-file requirements.txt

# Add new dependency
# 1. Add to pyproject.toml [project] dependencies or requirements.in
# 2. Recompile requirements.txt
# 3. pip install -r requirements.txt
```

## Configuration

### Required Environment Variables (.env file)
- `DISCORD_BOT_TOKEN` - Discord bot token
- `OPENAI_API_KEY` - OpenAI API key  
- `REPLYING_ALL_DISCORD_CHANNEL_IDS` - Comma-separated channel IDs where bot responds to all messages

### Optional Environment Variables  
- `ENABLE_PERSISTENCE=true` - Enable conversation persistence across restarts
- `GPT_ENGINE` - OpenAI model (default: gpt-5)
- `WEATHER_API_KEY`, `NEWS_API_KEY` - For weather/news commands
- `DISCORD_CHANNEL_ID` - Default channel for system messages
- `LOGGING=True` - Enable file logging
- `MESSAGE_REGEX` - Pattern for automatic responses in specified channels

## Testing

### Running Single Tests
```bash
# Run specific test file
pytest tests/test_conversation_manager.py

# Run specific test function
pytest tests/test_conversation_manager.py::test_channel_isolation

# Run with verbose output
pytest -v tests/
```

### Test Structure
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests with external APIs
- Use `@pytest.mark.slow` for tests that take >1 second
- Use `@pytest.mark.integration` for tests requiring external services

## Memory Management

### Conversation Contexts
- Contexts auto-cleanup after 7 days of inactivity
- Use `/context_cleanup` slash command for manual cleanup
- Important facts are preserved during cleanup if persistence is enabled
- Each context maintains: conversation history, learned preferences, important facts, user profiles

### Image Caching
- Discord images are cached as low-resolution base64 data URLs 
- Maximum 50 images cached with FIFO eviction
- Prevents issues with Discord CDN URL expiration

## Slash Commands

### Context Management
- `/context_info` - Show current channel's conversation stats
- `/context_clear` - Clear current channel's history (requires Manage Messages)
- `/context_list` - List all active contexts (admin only)
- `/context_export` - Export conversation as JSON
- `/context_cleanup` - Remove inactive contexts (admin only)

### Core Bot Commands
- `/chat [message]` - Direct chat with bot
- `/reset` - Clear current channel's conversation history
- `/draw [prompt]` - Generate images using DALL-E 3
- `/help` - Show available commands
- `/info` - Display bot configuration info

## Important Notes

### Action Code System
The bot uses action codes in responses to trigger real-time data fetching:
- `!WEATHER <location>` - Weather information
- `!SPORTS <query>` - Historical sports scores  
- `!SPORTSNOW <query>` - Live sports scores
- `!NEWS <topic>` - News articles
- `!DRAW <prompt>` - Image generation

### Message Processing
- Bot responds to mentions in any channel
- In configured channels (`REPLYING_ALL_DISCORD_CHANNEL_IDS`), responds to messages matching `MESSAGE_REGEX`  
- Always responds in DMs
- Duplicate message processing is prevented via message ID tracking

### Persistence Behavior
- When `ENABLE_PERSISTENCE=true`, conversations survive bot restarts
- Storage format is JSON files in `conversation_memories/` directory
- Important facts and preferences are preserved during context cleanup
- Future storage backends can be implemented via `MemoryStorageInterface`

<citations>
<document>
    <document_type>WARP_DOCUMENTATION</document_type>
    <document_id>getting-started/quickstart-guide/coding-in-warp</document_id>
</document>
</citations>
