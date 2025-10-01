from discord import Message
from utils.log import logger
import re


def smart_split_message(text: str, max_length: int = 2000) -> list[str]:
    """
    Smart message splitting that respects:
    - Code blocks (```, `)
    - Paragraphs
    - Sentences
    - Word boundaries

    Returns a list of message chunks.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    # Check if we're inside a code block
    def count_code_blocks(s: str) -> tuple[int, int]:
        """Count triple backticks and single backticks"""
        triple = s.count("```")
        single = s.count("`") - (triple * 3)
        return triple, single

    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    for para_idx, paragraph in enumerate(paragraphs):
        # Check if adding this paragraph would exceed the limit
        test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph

        if len(test_chunk) <= max_length:
            current_chunk = test_chunk
        else:
            # Current paragraph doesn't fit
            if current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = ""

            # If single paragraph is too long, split it
            if len(paragraph) > max_length:
                # Try splitting by sentences
                sentences = re.split(r'([.!?]+\s+)', paragraph)

                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]  # Add punctuation

                    test_chunk = current_chunk + sentence

                    if len(test_chunk) <= max_length:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            # Even single sentence is too long, split by words
                            words = sentence.split()
                            for word in words:
                                test_chunk = current_chunk + (" " if current_chunk else "") + word
                                if len(test_chunk) <= max_length:
                                    current_chunk = test_chunk
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = word
            else:
                current_chunk = paragraph

    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Verify no code blocks were broken
    # If a chunk ends with an odd number of code markers, carry them to next chunk
    fixed_chunks = []
    for i, chunk in enumerate(chunks):
        triple, single = count_code_blocks(chunk)

        # If we have unclosed triple backticks, add a closer
        if triple % 2 == 1 and i < len(chunks) - 1:
            chunk += "\n```"
            chunks[i + 1] = "```\n" + chunks[i + 1]

        # If we have unclosed single backticks, add a closer
        if single % 2 == 1 and i < len(chunks) - 1:
            chunk += "`"
            chunks[i + 1] = "`" + chunks[i + 1]

        fixed_chunks.append(chunk)

    return fixed_chunks


async def send_split_message(client, response: str, message):
    """Send a message, splitting it intelligently if necessary to fit Discord's length limits."""
    # Get the original channel to ensure consistency
    channel = getattr(message, 'channel', None)

    if len(response) <= 2000:
        # Check if this is an interaction or regular message
        if hasattr(message, 'followup'):
            await message.followup.send(response)
            logger.info(f"Sent response to interaction in channel {getattr(message, 'channel_id', 'unknown')}")
        else:
            await channel.send(response)
            logger.info(f"Sent response to message in channel {channel.id}")
        return

    # Smart split respecting formatting
    chunks = smart_split_message(response)

    logger.info(f"Splitting message into {len(chunks)} parts (total {len(response)} chars)")

    # Send each chunk
    for i, chunk in enumerate(chunks):
        if hasattr(message, 'followup'):
            await message.followup.send(chunk)
            if i == 0:
                logger.info(f"Sent split response part {i+1}/{len(chunks)} to interaction")
        else:
            await channel.send(chunk)
            if i == 0:
                logger.info(f"Sent split response part {i+1}/{len(chunks)} to channel {channel.id}")
