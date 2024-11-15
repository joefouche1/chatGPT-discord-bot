from discord import Message


async def send_split_message(client, response: str, message):
    """Send a message, splitting it if necessary to fit Discord's length limits."""
    if len(response) <= 2000:
        # Check if this is an interaction or regular message
        if hasattr(message, 'followup'):
            await message.followup.send(response)
        else:
            await message.channel.send(response)
        return

    # Split into chunks of 2000 characters
    chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]
    
    # Send each chunk
    for chunk in chunks:
        if hasattr(message, 'followup'):
            await message.followup.send(chunk)
        else:
            await message.channel.send(chunk)
