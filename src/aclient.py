import os
import discord
import asyncio

from src.log import logger

from utils.message_utils import send_split_message

from dotenv import load_dotenv
from discord import app_commands

import openai
import tiktoken

load_dotenv()

"""
This module contains the aclient class which is a subclass of discord.Client.
It handles the interaction with the Discord API and the OpenAI API.
"""


class aclient(discord.Client):
    """
    This class is a subclass of discord.Client and handles the interaction with the Discord API and the OpenAI API.
    It initializes the chatbot model and processes incoming messages.
    """
    def __init__(self) -> None:
        """
        Initializes the aclient class with default intents, command tree, activity, and environment variables.
        It also reads the system prompt from a file and initializes the chatbot model.
        """
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        self.current_channel = None
        self.activity = discord.Activity(type=discord.ActivityType.listening, name="bacon sizzle")
        
        self.replying_all_discord_channel_ids = set(int(id) for id in os.getenv("REPLYING_ALL_DISCORD_CHANNEL_IDS").split(','))

        self.openAI_API_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openAI_API_key
        self.openAI_gpt_engine = os.getenv("GPT_ENGINE")
        self.temperature = 0.75

        config_dir = os.path.abspath(f"{__file__}/../../")
        prompt_name = 'system_prompt.txt'
        prompt_path = os.path.join(config_dir, prompt_name)
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.starting_prompt = f.read()

        self.truncate_limit: int = (
            30500
            if "gpt-4-32k" in self.openAI_gpt_engine
            else 6500
            if "gpt-4" in self.openAI_gpt_engine
            else 14500
            if "gpt-3.5-turbo-16k" in self.openAI_gpt_engine
            else 3500
        )

        self.conversation_history = None
        self.history_lock = asyncio.Lock()
        self.init_history()
        self.message_queue = asyncio.Queue()

    def init_history(self):
        """
        Initializes the conversation history with a system prompt.
        """
        self.conversation_history = [
            {
                "role": "system",
                "content": self.starting_prompt
            }
        ]

    async def ainit_history(self):
        """
        Asynchronously initializes the conversation history with a system prompt.
        """
        await self.init_history()

    async def handle_response(self, message: discord.Message = None) -> str:
        """
        Handles the response from the chatbot model and adds it to the conversation history.
        """
        response = ""
        spoke = False
        spoke2 = False

        async_generator = self.ask_stream_async(message)
        try:
            while True:
                try:
                    token = await asyncio.wait_for(async_generator.__anext__(), timeout=40)
                    response += token
                    if len(response) > 20 and not spoke:
                        logger.info(f"Response has started with: {response}")
                        spoke = True
                    elif len(response) % 500 > 0 and spoke and (not spoke2):
                        logger.info(f"Response is up to {len(response)}")
                        spoke2 = True
                except StopAsyncIteration:
                    # This exception is raised when the async generator is exhausted
                    break
        except asyncio.TimeoutError:
            logger.warning("handle_response took over 40 seconds with no reply.")
        # Add the model's response to the conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        return response

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        tiktoken.model.MODEL_TO_ENCODING["gpt-4"] = "cl100k_base"
        encoding = tiktoken.encoding_for_model(self.openAI_gpt_engine)
        num_tokens = 0
        # for message in self.conversation[convo_id]:
        for message in self.conversation_history:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 5
            for key, value in message.items():
                if value:
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 5  # role is always required and always 1 token
        num_tokens += 5  # every reply is primed with <im_start>assistant
        return num_tokens

    async def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (self.get_token_count(convo_id) > self.truncate_limit and len(self.conversation_history) > 1):
                # Don't remove the first message
                async with self.history_lock:
                    self.conversation_history.pop(1)
                logger.info("popped an old message!")
            else:
                break

    async def ask_stream_async(self, message=None):
        """
        Adds the user's message to the conversation history and initializes the chat models with the conversation history.
        """
        engine = self.openAI_gpt_engine
        if message:
            # Check if the current message contains the tag *GPT4*
            if "*GPT4*" in message:
                engine = 'gpt-4'
                # Remove the gpt4 hint from the content
                logger.info("Using GPT4 for current activity")
                message = message.replace("*GPT4*", "") 
        
            # Acquire the lock before modifying the conversation_history
            async with self.history_lock:
                # Add the user's message to the conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": message
                })

        await self.__truncate_conversation()

        if self.conversation_history is None:
            logger.warning("No history is set!")

        # Initialize the chat models with the conversation history
        completion = await openai.ChatCompletion.acreate(
            model=engine,
            messages=self.conversation_history,
            temperature=self.temperature,
            stream=True
        )

        async for chunk in completion:
            dd = chunk.choices[0].delta
            if 'content' in dd:
                yield dd.content
            else:
                break

    async def process_messages(self):
        """
        Processes incoming messages and sends them to the chatbot model for a response.
        """
        while True:
            if self.current_channel is not None:
                while not self.message_queue.empty():
                    await self.current_channel.typing()
                    message, user_message = await self.message_queue.get()
                    try:
                        await self.send_message(message, user_message)
                    except Exception as e:
                        logger.exception(f"Error while processing message: {e}")
                    finally:
                        self.message_queue.task_done()
            await asyncio.sleep(1)

    async def enqueue_message(self, message: discord.Message, user_message):
        """
        Adds a message to the message queue for processing.
        """
        await message.response.defer(ephemeral=self.isPrivate) if self.is_replying_all == "False" else None
        await self.message_queue.put((message, user_message))

    async def send_message(self, message: discord.Message, user_message):
        """
        Sends a message to the chatbot model and handles the response.
        """
        author = message.author.id
        try:
            normal_reply = ("witty reaction" not in user_message)
            if normal_reply:
                response = (f'> **{user_message}** - <@{str(author)}> \n\n')
            else:
                response = ""

            model_out = await self.handle_response(user_message)
            response = f"{response}{model_out}"
            logger.info(f"Response complete [size {len(model_out)}]")

            if normal_reply:
                await send_split_message(self, response, message)
            else:
                await message.reply(model_out, mention_author=False)

        except Exception as e:
            logger.exception(f"Error while sending : {e}")
            await message.channel.send(f"> **ERROR: Something went wrong, please try again later!** \n ```ERROR MESSAGE: {e}```")

    async def send_start_prompt(self):
        """
        Sends the starting prompt to the chatbot model and handles the response.
        """
        discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
        try:
            if self.starting_prompt:
                if (discord_channel_id):
                    channel = self.get_channel(int(discord_channel_id))
                    logger.info(f"Send system prompt with size {len(self.starting_prompt)}")
                    response = await self.handle_response()
                    if len(response) > 0:
                        await channel.send(response)
                    logger.info(f"System prompt response:{response}")
                else:
                    logger.info("No Channel selected. Skip sending system prompt.")
            else:
                logger.info("Not given starting prompt. Skipping...")
        except Exception as e:
            logger.exception(f"Error while sending system prompt: {e}")
