from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def test_api():
    client = AsyncOpenAI()

    stream = await client.responses.create(
        model="gpt-5",
        stream=True,
        max_output_tokens=1000,
        instructions="You are a helpful assistant.",
        input="Oink oink! What's the diameter of the average avocado?",
        reasoning={"effort": "low"},
        parameters={"verbosity": "low"}
    )

    async for event in stream:
        if event.type == 'content.text.delta':
            print(event.text, end='', flush=True)
        elif event.type == 'content.text.done':
            print()  # Newline at the end

# Create an event loop
loop = asyncio.get_event_loop()

# Run the test_api method in the event loop
loop.run_until_complete(test_api())
