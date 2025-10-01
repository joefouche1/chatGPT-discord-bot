# Multi-Context Conversation Guide

## Overview

The bot now supports **per-channel conversation contexts**, allowing it to maintain separate conversation histories for each Discord channel. This means the bot can carry on different conversations simultaneously in different channels without mixing them up.

## Key Features

### 1. Per-Channel Conversation Isolation
- Each channel has its own conversation history
- Conversations in one channel don't affect others
- The bot remembers context specific to each channel

### 2. Persistent Memory Support (Optional)
- Enable persistence by setting `ENABLE_PERSISTENCE=true` in your `.env` file
- Conversations are saved to disk and survive bot restarts
- Important facts and preferences can be stored for future reference

### 3. Context Management Commands

The bot includes several slash commands for managing conversation contexts:

#### `/context_info`
- Shows information about the current channel's conversation
- Displays message count, creation time, last activity
- Shows learned preferences and important facts (if any)

#### `/context_clear`
- Clears the conversation history for the current channel
- Requires "Manage Messages" permission
- Useful for starting fresh conversations

#### `/context_list` (Admin only)
- Lists all active conversation contexts across the server
- Shows message counts and last activity times
- Helps administrators monitor bot usage

#### `/context_export`
- Exports the current channel's conversation history as JSON
- Requires "Manage Messages" permission
- Useful for backing up conversations or analysis

#### `/context_cleanup` (Admin only)
- Removes conversation contexts inactive for more than 7 days
- Helps manage memory usage
- Preserves important memories before cleanup

#### `/reset`
- Now only resets the current channel's conversation
- No longer affects other channels

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Enable persistent storage (optional)
ENABLE_PERSISTENCE=false

# For startup messages (optional)
DISCORD_CHANNEL_ID=your_default_channel_id
```

### Persistence Storage

When persistence is enabled:
- Conversations are stored in `conversation_memories/` directory
- Each channel gets its own JSON file
- Important memories are preserved separately

## Architecture

### Core Components

1. **EnhancedConversationManager**
   - Manages all channel contexts
   - Handles persistence if enabled
   - Provides cleanup and statistics

2. **ChannelConversationContext**
   - Individual context for each channel
   - Tracks conversation history
   - Stores metadata and learned preferences

3. **MemoryStorageInterface**
   - Abstract interface for storage backends
   - Currently implements JSON file storage
   - Extensible for future database backends

### Data Structure

Each channel context maintains:
- `conversation_history`: List of messages
- `metadata`: Creation time, last activity, message count
- `learned_preferences`: Channel-specific preferences
- `important_facts`: Notable information to remember
- `user_profiles`: Information about users in the channel

## Future Enhancements

The system is designed for future expansion:

### Planned Features
1. **Smart Memory Extraction**
   - Automatically identify important facts
   - Learn user preferences over time
   - Build user profiles automatically

2. **Database Storage**
   - PostgreSQL/MySQL support
   - Redis for fast caching
   - Cloud storage integration

3. **Memory Sharing**
   - Share learned facts across channels (with permission)
   - Global memories that apply everywhere
   - User-specific memories that follow users

4. **Advanced Context Management**
   - Context templates for different channel types
   - Context inheritance for thread channels
   - Context merging and splitting

## Usage Examples

### Basic Conversation
```
Channel #general:
User: Tell me about Python
Bot: [Responds about Python]

Channel #gaming:
User: Tell me about Python
Bot: [Can respond differently, doesn't mix with #general context]
```

### With Persistence Enabled
```
Day 1 - Channel #project:
User: Remember that our deadline is Friday
Bot: I'll remember that the deadline is Friday.

[Bot restarts]

Day 2 - Channel #project:
User: What's our deadline?
Bot: The deadline is Friday, as you mentioned earlier.
```

### Context Management
```
# Check current context
/context_info

# Clear history for fresh start
/context_clear

# Export conversation for records
/context_export
```

## Migration from Single Context

If you're upgrading from the single-context version:
1. The bot will automatically create new contexts for each channel
2. Old conversation history won't be migrated (starts fresh)
3. No configuration changes required for basic operation
4. Enable persistence if you want conversations to survive restarts

## Performance Considerations

- Each context uses memory proportional to its conversation length
- Automatic truncation prevents excessive memory usage
- Inactive contexts can be cleaned up automatically
- Token counting is now per-channel for accurate limits

## Troubleshooting

### Bot Not Remembering Conversations
- Check if persistence is enabled in `.env`
- Verify the `conversation_memories/` directory has write permissions
- Check logs for any storage errors

### Memory Usage High
- Run `/context_cleanup` to remove old contexts
- Consider reducing `max_inactive_hours` in configuration
- Use `/context_clear` in channels with long histories

### Contexts Not Isolating
- Ensure you're using the latest version
- Check that channel IDs are being properly detected
- Review logs for any context switching errors

## Development

### Adding New Storage Backends

Implement the `MemoryStorageInterface`:

```python
class MyStorageBackend(MemoryStorageInterface):
    async def save_context(self, channel_id: str, context_data: dict):
        # Your implementation
        pass
    
    async def load_context(self, channel_id: str) -> Optional[dict]:
        # Your implementation
        pass
    # ... etc
```

### Extending Context Data

Add new fields to `ChannelConversationContext`:

```python
class ChannelConversationContext:
    def __init__(self, ...):
        # Existing fields
        self.custom_data = {}  # Your new field
    
    def add_custom_data(self, key: str, value: Any):
        self.custom_data[key] = value
```

## Best Practices

1. **Regular Cleanup**: Run `/context_cleanup` weekly to manage memory
2. **Export Important Conversations**: Use `/context_export` for valuable discussions
3. **Clear When Needed**: Don't hesitate to use `/context_clear` for fresh starts
4. **Monitor Usage**: Use `/context_list` to track active conversations
5. **Enable Persistence**: For production bots, enable persistence for better UX

## Support

For issues or questions about the multi-context system:
1. Check the logs in your console for detailed error messages
2. Ensure all dependencies are up to date
3. Verify your `.env` configuration
4. Test with a simple conversation first
5. Use `/context_info` to debug context state
