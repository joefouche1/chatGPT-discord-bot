
from revChatGPT.V3 import Chatbot

import openai, os 
from dotenv import load_dotenv

load_dotenv()

mybot = Chatbot(api_key=os.getenv("OPENAI_API_KEY"), engine="gpt-3.5-turbo", system_prompt="you are a helpful assistant, but answer sarcastically.")

async def ask_and_print():
    async for token in mybot.ask_stream_async("What is the diameter of an ozone molecule?"):
        print(token)

# Run the async function
import asyncio
asyncio.run(ask_and_print())