import os
import discord
import asyncio
import aiohttp
import io

from utils.log import logger

from utils.message_utils import send_split_message

from dotenv import load_dotenv

from discord import app_commands

from openai import AsyncOpenAI


import tiktoken

from aiohttp import ClientSession
from discord.ext import commands
from discord import Embed


# Check for action codes in the response before returning
from src.commands.actions import process_action_code
from src.commands.weather import get_weather
from src.commands.news import get_news
from src.commands.sports import get_sports_score
        

load_dotenv()

"""
This module contains the aclient class which is a subclass of discord.Client.
It handles the interaction with the Discord API and the OpenAI API.
"""


class ConversationManager:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, message):
        self.conversation_history.append(message)

    def get_history(self):
        return self.conversation_history


class aclient(discord.Client):
    """
    This class is a subclass of discord.Client and handles the interaction with the Discord API and the OpenAI API.
    It initializes the chatbot model and processes incoming messages.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the aclient class with default intents, command tree, activity, and environment variables.
        It also reads the system prompt from a file and initializes the chatbot model.
        """
        intents = discord.Intents.default()
        intents.message_content = True
        #intents.members = True
        super().__init__(intents=intents, **kwargs)
        self.conversation_manager = ConversationManager()

        self.tree = app_commands.CommandTree(self)
        self.current_channel = None
        self.activity = discord.Activity(
            type=discord.ActivityType.listening, name="bacon sizzle")

        self.replying_all_discord_channel_ids = set(
            int(id) for id in os.getenv("REPLYING_ALL_DISCORD_CHANNEL_IDS").split(','))

        self.openAI_API_key = os.getenv("OPENAI_API_KEY")
        self.openAI_gpt_engine = os.getenv("GPT_ENGINE")
        self.temperature = 0.75
        self.client = AsyncOpenAI()

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

        self.rate_limiter = commands.CooldownMapping.from_cooldown(1, 5.0, commands.BucketType.user)

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
                    if token is not None:
                        response += token
                        if len(response) > 0 and not spoke:
                            logger.info(
                                f"Response has started with: {response}")
                            spoke = True
                        elif len(response) % 500 > 0 and spoke and (not spoke2):
                            logger.info(f"Response is up to {len(response)}")
                            spoke2 = True
                except StopAsyncIteration:
                    break
        except asyncio.TimeoutError:
            logger.warning(
                "handle_response took over 40 seconds with no reply.")

        # Add the model's response to the conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        try:
            action_result = await process_action_code(response)
            if action_result:
                code, params = action_result
                #logger.info(f"Found action code {code} with params {params}")
                
                # Execute the action and get any additional response
                if code == "!WEATHER":
                    await get_weather(self.current_channel, params)
                    return None
                elif code == "!SPORTS":
                    try:
                        result = await get_sports_score(params, self.client)
                        if result:
                            from src.commands.sports import format_sports_response
                            embed = await format_sports_response(result)
                            await self.current_channel.send(embed=embed)
                        return None
                    except Exception as e:
                        await self.current_channel.send(f"Sorry, I couldn't find that score. Error: {str(e)}")
                        return None
                elif code == "!NEWS":
                    await get_news(self.current_channel, params)
                    return None
                elif code == "!DRAW":
                    try:
                        image_url = await self.draw(params)
                        
                        # Download the image
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_url) as resp:
                                image_data = await resp.read()

                        # Create a discord.File object
                        image_file = discord.File(io.BytesIO(
                            image_data), filename="image.png")

                        # Create an Embed object that includes the image
                        embed = discord.Embed(description=params, title="AI hog generated image")
                        embed.set_image(url="attachment://image.png")

                        # Send the embed and the image as an attachment
                        await self.current_channel.send("Here you go!", embed=embed, file=image_file)
                        return None
                    except Exception as e:
                        error_msg = str(e)
                        if 'content_policy_violation' in error_msg:
                            embed = discord.Embed(
                                title="🚫 Oink Oink! Content Policy Violation",
                                description=f"Your prompt: `{params}`",
                                color=discord.Color.red()
                            )
                            embed.add_field(
                                name="Description",
                                value="The AI pig has detected some questionable content! Let's keep things family-friendly.",
                                inline=False
                            )
                            await self.current_channel.send(embed=embed)
                        else:
                            await self.current_channel.send(f'> **Something Went Wrong: {e}**')
                        return None
                
        except Exception as e:
            logger.error(f"Error processing action code in model response: {e}")

        return response

    async def handle_image_attachment(self, message: discord.Message, user_message: str) -> str:
        """
            Attempt to deal with an attached picture as a special request

            Discord urls can be resized.
        """
        img_url = message.attachments[0].url + "?width=500&height=500"

        logger.info(f"Prompting with photo at {img_url}")
        # Craft the prompt for GPT
        prompt_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        }
                    }
                ]
            }
        ]
        newhist = self.conversation_history + prompt_message

        # Initialize the chat models with the conversation history
        completion = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=newhist,
            temperature=self.temperature,
            max_tokens=2000
        )
        return completion.choices[0].message.content

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
        Adds the user's message to the conversation history and initializes
        the chat models with the conversation history.
        """
        engine = self.openAI_gpt_engine
        if message:
            # Check if the current message contains the tag *GPT4*
            # if "*GPT4*" in message:
            #     engine = 'gpt-4'
            #     # Remove the gpt4 hint from the content
            #     logger.info("Using GPT4 for current activity")
            #     message = message.replace("*GPT4*", "")

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
        stream = await self.client.chat.completions.create(model=engine,
                                                           messages=self.conversation_history,
                                                           temperature=self.temperature,
                                                           max_tokens=4000,
                                                           stream=True)

        async for completion in stream:
            # print(completion.model_dump_json())
            dd = completion.choices[0].delta
            if dd.content is not None:
                yield dd.content

    async def process_messages(self):
        while True:
            # This will block until a message is available
            message, user_message = await self.message_queue.get()
            await self.send_message(message, user_message)
            # Mark the task as done
            self.message_queue.task_done()

    async def enqueue_message(self, message, user_message):
        """
        Add a message to the processing queue
        """
        if hasattr(message, 'response'):  # This is an Interaction
            await message.response.defer(ephemeral=self.isPrivate) if self.is_replying_all == "False" else None
        
        await self.message_queue.put((message, user_message))

    async def send_message(self, message: discord.Message, user_message):
        """
        Sends a message to the chatbot model and handles the response.
        """
        author = message.author.id
        try:
            # Check if the message has any attachments
            if message.attachments:
                # If there are attachments, call a different handler
                img_reply = await self.handle_image_attachment(message, user_message)
                logger.info(f"Response complete [size {len(img_reply)}]")
                await send_split_message(self, img_reply, message)

            else:
                normal_reply = ("witty reaction" not in user_message)
                if normal_reply:
                    response_prefix = f'> **{user_message}** - <@{str(author)}> \n\n'
                else:
                    response_prefix = ""

                model_out = await self.handle_response(user_message)
                
                # Only send a message if model_out is not None
                if model_out is not None:
                    response = f"{response_prefix}{model_out}"
                    logger.info(f"Response complete [size {len(model_out)}]")

                    if normal_reply:
                        await send_split_message(self, response, message)
                    else:
                        await message.reply(model_out, mention_author=False)

        except Exception as e:
            logger.exception(f"Error while sending : {e}")
            await message.channel.send(f"> **ERROR: Something went wrong.** \n ```ERROR MESSAGE: {e}```")

    async def draw(self, prompt) -> list[str]:

        response = await self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard",
        )
        image_url = response.data[0].url

        return image_url

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
                    logger.info(
                        "No Channel selected. Skip sending system prompt.")
            else:
                logger.info("Not given starting prompt. Skipping...")
        except Exception as e:
            logger.exception(f"Error while sending system prompt: {e}")

    async def get_chat_response(self, message: str) -> str:
        """
        Gets a simple chat response without streaming.
        """
        # Add the user's message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        await self.__truncate_conversation()

        # Get completion from OpenAI
        completion = await self.client.chat.completions.create(
            model=self.openAI_gpt_engine,
            messages=self.conversation_history,
            temperature=self.temperature,
            max_tokens=1000
        )

        response = completion.choices[0].message.content

        # Add the response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    async def on_command(self, ctx):
        bucket = self.rate_limiter.get_bucket(ctx.message)
        retry_after = bucket.update_rate_limit()
        if retry_after:
            await ctx.send(f"You're doing that too much. Try again in {retry_after:.2f}s.")
            return True
        return False
