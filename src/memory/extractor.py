"""
Phase 2: Memory Extraction System

Automatically extracts memorable information from conversations using LLM analysis.
Identifies facts, preferences, decisions, and events worth remembering long-term.
"""

import logging
from typing import Optional, Tuple
from datetime import datetime

from .models import MemoryEntry, MemoryType, MemoryConfidence

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """
    Extracts memorable information from conversation exchanges using LLM analysis.
    
    Only captures HIGH confidence extractions to avoid noise in long-term memory.
    Integrates with HybridMemoryStorage to persist extracted memories.
    """
    
    # Extraction prompt template
    EXTRACTION_PROMPT = """Analyze this conversation exchange. Should any of this be remembered long-term?

User: {user_message}
Assistant: {assistant_message}

Extract memorable information ONLY if the user shared:
- **Facts**: Personal information (location, job, family, interests, technical details)
- **Preferences**: Communication style, likes/dislikes, how they want responses
- **Decisions**: Important choices, plans, commitments they've made
- **Events**: Scheduled activities, deadlines, travel plans, appointments

**Be conservative** - only extract truly important information worth remembering.

**Rules:**
1. Focus on USER information (not assistant's responses)
2. Extract concrete, specific details (not vague statements)
3. Avoid extracting casual conversation or transient topics
4. Only mark as HIGH confidence if explicitly stated by user
5. If nothing memorable, respond with TYPE: none

**Format your response EXACTLY as:**
TYPE: [fact|preference|decision|event|none]
CONTENT: [what to remember - be concise and specific]
CONFIDENCE: [high|medium|low]

**Examples:**

User: "I live in Pittsburgh and work as a software engineer"
Assistant: "Great! Pittsburgh has a vibrant tech scene..."
→ TYPE: fact
→ CONTENT: User lives in Pittsburgh, works as software engineer
→ CONFIDENCE: high

User: "I prefer concise answers without too much explanation"
Assistant: "Understood, I'll keep responses brief..."
→ TYPE: preference
→ CONTENT: Prefers concise, brief responses without extensive explanations
→ CONFIDENCE: high

User: "I'm flying to Vegas on Feb 5th for a conference"
Assistant: "Have a great trip! What conference?"
→ TYPE: event
→ CONTENT: Flying to Vegas on Feb 5th for a conference
→ CONFIDENCE: high

User: "What's the weather like?"
Assistant: "It's sunny and 72°F..."
→ TYPE: none
→ CONTENT: 
→ CONFIDENCE: 

Now analyze the conversation above:"""

    def __init__(self, openai_client):
        """
        Initialize the memory extractor.
        
        Args:
            openai_client: OpenAI client instance for LLM calls
        """
        self.client = openai_client
    
    async def extract_memory(
        self,
        user_message: str,
        assistant_message: str,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None
    ) -> Optional[MemoryEntry]:
        """
        Analyze a conversation exchange and extract memorable information.
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            channel_id: Discord channel ID (optional)
            user_id: Discord user ID (optional)
            username: Discord username (optional)
        
        Returns:
            MemoryEntry if extraction was successful and HIGH confidence, None otherwise
        """
        try:
            # Format the extraction prompt
            prompt = self.EXTRACTION_PROMPT.format(
                user_message=user_message[:1000],  # Limit length to avoid token overflow
                assistant_message=assistant_message[:1000]
            )
            
            # Call LLM for extraction
            logger.debug(f"Sending memory extraction request for channel {channel_id}")
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use fast, cheap model for extraction
                messages=[{
                    "role": "system",
                    "content": "You are a memory extraction system. Be conservative - only capture truly important information. Follow the format exactly."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=500
            )
            
            # Parse the response
            extraction_text = response.choices[0].message.content.strip()
            logger.debug(f"Memory extraction response: {extraction_text}")
            
            # Parse the structured output
            memory_type, content, confidence = self._parse_extraction(extraction_text)
            
            # Only return if HIGH confidence and valid type
            if confidence != MemoryConfidence.HIGH or memory_type is None:
                logger.debug(f"Skipping memory extraction - confidence: {confidence}, type: {memory_type}")
                return None
            
            # Create memory entry
            metadata = {}
            if username:
                metadata['username'] = username
            
            memory = MemoryEntry(
                content=content,
                memory_type=memory_type,
                channel_id=channel_id,
                user_id=user_id,
                timestamp=datetime.now(),
                confidence=confidence,
                metadata=metadata
            )
            
            logger.info(f"Extracted {memory_type.value} memory: {content[:100]}...")
            return memory
            
        except Exception as e:
            logger.error(f"Error extracting memory: {e}", exc_info=True)
            return None
    
    def _parse_extraction(self, text: str) -> Tuple[Optional[MemoryType], str, MemoryConfidence]:
        """
        Parse the LLM extraction output into structured components.
        
        Args:
            text: Raw LLM response text
        
        Returns:
            Tuple of (MemoryType, content, MemoryConfidence)
        """
        lines = text.strip().split('\n')
        
        memory_type = None
        content = ""
        confidence = MemoryConfidence.LOW
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('TYPE:'):
                type_str = line.replace('TYPE:', '').strip().lower()
                if type_str == 'none':
                    return None, "", MemoryConfidence.LOW
                try:
                    memory_type = MemoryType(type_str)
                except ValueError:
                    logger.warning(f"Invalid memory type: {type_str}")
                    return None, "", MemoryConfidence.LOW
            
            elif line.startswith('CONTENT:'):
                content = line.replace('CONTENT:', '').strip()
            
            elif line.startswith('CONFIDENCE:'):
                conf_str = line.replace('CONFIDENCE:', '').strip().lower()
                try:
                    confidence = MemoryConfidence(conf_str)
                except ValueError:
                    logger.warning(f"Invalid confidence level: {conf_str}")
                    confidence = MemoryConfidence.LOW
        
        # Validate we got required fields
        if not memory_type or not content:
            logger.debug(f"Incomplete extraction - type: {memory_type}, content: {content}")
            return None, "", MemoryConfidence.LOW
        
        return memory_type, content, confidence


async def extract_and_store_memory(
    extractor: MemoryExtractor,
    storage,
    user_message: str,
    assistant_message: str,
    channel_id: Optional[str] = None,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    channel_name: Optional[str] = None,
    guild_name: Optional[str] = None
):
    """
    Helper function to extract and store a memory in one call.
    
    Args:
        extractor: MemoryExtractor instance
        storage: HybridMemoryStorage instance
        user_message: The user's message
        assistant_message: The assistant's response
        channel_id: Discord channel ID (optional)
        user_id: Discord user ID (optional)
        username: Discord username (optional)
        channel_name: Discord channel name (optional)
        guild_name: Discord guild name (optional)
    """
    # Extract memory
    memory = await extractor.extract_memory(
        user_message=user_message,
        assistant_message=assistant_message,
        channel_id=channel_id,
        user_id=user_id,
        username=username
    )
    
    if memory:
        # Store in daily log
        await storage.save_daily_note(
            memory=memory,
            channel_name=channel_name,
            guild_name=guild_name
        )
        
        # Store in long-term memory
        await storage.add_long_term_memory(memory)
        
        logger.info(f"Stored memory: [{memory.memory_type.value}] {memory.content[:100]}...")
