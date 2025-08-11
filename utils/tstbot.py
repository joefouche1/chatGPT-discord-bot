
from openai import AsyncOpenAI
import os 
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Initialize the client with the official SDK
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def ask_and_print():
    # Use the Responses API with streaming
    stream = await client.responses.create(
        model="gpt-5",
        instructions="You are a helpful assistant, but answer sarcastically.",
        input="What is the diameter of an ozone molecule?",
        stream=True,
        max_output_tokens=1000
    )
    
    # Process streaming events
    async for event in stream:
        if event.type == 'content.text.delta':
            print(event.text, end='', flush=True)
        elif event.type == 'content.text.done':
            print()  # Newline at the end

# Run the async function
asyncio.run(ask_and_print())