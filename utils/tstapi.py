from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def test_api():
    client = AsyncOpenAI()

    stream = await client.chat.completions.create(model="gpt-5",
                                                  stream=True,
                                                  max_completion_tokens=1000,
                                                  messages=[{"role": "user", "content": "Oink oink! What's the diameter of the average avocado?"}])

    async for completion in stream:
        print(completion.model_dump_json())

# Create an event loop
loop = asyncio.get_event_loop()

# Run the test_api method in the event loop
loop.run_until_complete(test_api())
