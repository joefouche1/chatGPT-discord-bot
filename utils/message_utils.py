from discord import Message
from utils.log import logger


async def send_split_message(client, response: str, message):
    """Send a message, splitting it if necessary to fit Discord's length limits."""
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

    # Split into chunks of 2000 characters
    chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]
    
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
