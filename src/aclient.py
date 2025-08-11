import os
import discord
import asyncio
import aiohttp
import io
import re
import urllib.parse
import base64
from PIL import Image

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


class aclient(commands.Bot):
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
        super().__init__(command_prefix="!", intents=intents, **kwargs)
        self.conversation_manager = ConversationManager()

        # Use the built-in command tree from commands.Bot (already initialized by super().__init__)
        self.current_channel = None
        self.activity = discord.Activity(
            type=discord.ActivityType.listening, name="bacon sizzle")

        self.replying_all_discord_channel_ids = set(
            int(id) for id in os.getenv("REPLYING_ALL_DISCORD_CHANNEL_IDS").split(','))

        self.openAI_API_key = os.getenv("OPENAI_API_KEY")
        self.openAI_gpt_engine = os.getenv("GPT_ENGINE", "gpt-5")
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
        self.system_prompt_sent = False  # Track if system prompt has been sent

        self.rate_limiter = commands.CooldownMapping.from_cooldown(1, 5.0, commands.BucketType.user)
        
        # Image cache to store low-resolution versions of Discord images
        self.image_cache = {}  # {original_url: base64_data_url}
        self.max_cache_size = 50  # Maximum number of images to cache

    def init_history(self):
        """
        Initializes the conversation history without system prompt (will use instructions parameter instead).
        """
        self.conversation_history = []

    async def ainit_history(self):
        """
        Asynchronously initializes the conversation history with a system prompt.
        """
        await self.init_history()

    def _format_history_for_responses(self):
        """
        Format conversation history for the Responses API.
        Combines all messages into a single input string.
        """
        if not self.conversation_history:
            return ""
        
        formatted_parts = []
        for msg in self.conversation_history:
            role = msg.get("role")
            content = msg.get("content")
            
            if isinstance(content, str):
                if role == "user":
                    formatted_parts.append(f"User: {content}")
                elif role == "assistant":
                    formatted_parts.append(f"Assistant: {content}")
            elif isinstance(content, list):
                # Handle multimodal content
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                if text_parts:
                    combined_text = " ".join(text_parts)
                    if role == "user":
                        formatted_parts.append(f"User: {combined_text}")
                    elif role == "assistant":
                        formatted_parts.append(f"Assistant: {combined_text}")
        
        return "\n\n".join(formatted_parts)

    async def cache_discord_image(self, image_url: str) -> str:
        """
        Downloads and caches a Discord image as a low-resolution base64 data URL.
        Returns the cached data URL to avoid CDN expiration issues.
        """
        # Check if already cached
        if image_url in self.image_cache:
            return self.image_cache[image_url]
        
        try:
            # Download the image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to download image: {resp.status}")
                        return image_url  # Fallback to original URL
                    
                    image_data = await resp.read()
            
            # Open and resize the image to low resolution
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary (to handle RGBA images)
            if img.mode in ('RGBA', 'LA'):
                # Create a white background
                bg = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    bg.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                else:
                    bg.paste(img)
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to low resolution to save memory (max 400px on longest side)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Convert to base64 data URL
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data_url = f"data:image/jpeg;base64,{img_str}"
            
            # Cache the result with cache size management
            if len(self.image_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
                logger.info(f"Removed oldest cached image to make space")
            
            self.image_cache[image_url] = data_url
            logger.info(f"Cached image with size: {len(img_str)} characters")
            
            return data_url
            
        except Exception as e:
            logger.error(f"Error caching image {image_url}: {e}")
            return image_url  # Fallback to original URL

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

        # Check for LaTeX expressions and process them
        if response and "\\begin{" in response or "\\frac" in response or "\\text" in response:
            # If we find LaTeX, return a special marker to indicate it needs image processing
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            return f"__LATEX_RENDER__{response}"
        
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
            Handle processing of image attachments and add them to conversation history cleanly.
        """
        original_img_url = message.attachments[0].url + "?width=500&height=500"

        logger.info(f"Prompting with photo at {original_img_url}")
        
        # Cache the image to avoid CDN expiration issues
        cached_img_url = await self.cache_discord_image(original_img_url)
        
        # Append the image + text as a single user message into the persistent history
        # Use cached image URL to prevent CDN expiration
        async with self.history_lock:
            self.conversation_history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": cached_img_url}}
                ]
            })

        await self.__truncate_conversation()

        # Format multimodal input for Responses API
        # Use cached image URL for API calls as well
        multimodal_input = [
            {"type": "input_text", "text": self._format_history_for_responses()},
            {"type": "input_text", "text": f"\n\nUser: {user_message}"},
            {"type": "input_image", "image_url": cached_img_url}
        ]

        # Use GPT-5 for image understanding while preserving context
        completion = await self.client.responses.create(
            model="gpt-5",
            instructions=self.starting_prompt,
            input=multimodal_input,
            temperature=self.temperature,
            max_output_tokens=10000
        )

        reply = completion.output

        # Add assistant's reply to history
        async with self.history_lock:
            self.conversation_history.append({
                "role": "assistant",
                "content": reply
            })

        return reply

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        try:
            # Try to get the encoding for the model
            encoding = tiktoken.encoding_for_model(self.openAI_gpt_engine)
        except KeyError:
            # If model not found, use cl100k_base as default
            logger.info(f"Model {self.openAI_gpt_engine} not found in tiktoken mappings, using cl100k_base encoding")
            encoding = tiktoken.get_encoding("cl100k_base")
            
        num_tokens = 0
        # for message in self.conversation[convo_id]:
        for message in self.conversation_history:
            # base overhead per message
            num_tokens += 5

            role = message.get("role")
            if role:
                num_tokens += len(encoding.encode(role))

            content = message.get("content")
            if isinstance(content, str):
                num_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type == "text":
                        text_value = part.get("text", "")
                        if text_value:
                            num_tokens += len(encoding.encode(text_value))
                    elif part_type == "image_url":
                        # Rough overhead for an image reference; images don't consume text tokens
                        num_tokens += 10
                    else:
                        # Fallback: ignore unknown parts
                        pass
            elif isinstance(content, dict):
                # Rare case: serialize keys minimally
                for k, v in content.items():
                    if isinstance(v, str):
                        num_tokens += len(encoding.encode(v))
            # name handling
            if "name" in message:
                num_tokens += 5
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
        stream = await self.client.responses.create(
            model=engine,
            instructions=self.starting_prompt,
            input=self._format_history_for_responses(),
            temperature=self.temperature,
            max_output_tokens=4000,
            stream=True
        )

        async for event in stream:
            if event.type == 'content.text.delta':
                yield event.text

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
        # Ensure we're explicitly using the message's channel rather than current_channel
        response_channel = message.channel
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
                    # Check if the response contains LaTeX that needs rendering
                    if model_out.startswith("__LATEX_RENDER__"):
                        # Extract the actual content
                        latex_content = model_out[16:]
                        # Generate a text-only version with removed LaTeX commands for the message
                        text_content = re.sub(r'\\begin\{.*?\}|\\end\{.*?\}|\\text\{(.*?)\}', r'\1', latex_content)
                        text_content = re.sub(r'\\[a-zA-Z]+', '', text_content)
                        text_content = re.sub(r'\\\[|\\\]|\\\(|\\\)|_|\^|&', '', text_content)
                        
                        # Create the response message with the original user message
                        response = f"{response_prefix}{text_content}\n\n*Mathematical expressions rendered as images:*"
                        
                        # Send the text part
                        await send_split_message(self, response, message)
                        
                        # Render the LaTeX expressions
                        image_urls = await self.render_latex(latex_content)
                        if image_urls:
                            for url in image_urls:
                                # Download the image
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(url) as resp:
                                        if resp.status == 200:
                                            image_data = await resp.read()
                                            # Create a discord.File object
                                            image_file = discord.File(io.BytesIO(image_data), filename="math.png")
                                            # Send the image
                                            await response_channel.send(file=image_file)
                    else:
                        response = f"{response_prefix}{model_out}"
                        logger.info(f"Response complete [size {len(model_out)}] to channel {response_channel.id}")

                    if normal_reply:
                        await send_split_message(self, response, message)
                    else:
                        await message.reply(model_out, mention_author=False)

        except Exception as e:
            logger.exception(f"Error while sending : {e}")
            await response_channel.send(f"> **ERROR: Something went wrong.** \n ```ERROR MESSAGE: {e}```")

    async def draw(self, prompt) -> list[str]:
        """Generate image using DALL-E model"""
        try:
            # Use GPT-5 to refine/improve the prompt
            refine_response = await self.client.responses.create(
                model="gpt-5",
                instructions="You are a helpful image prompt engineer. Your task is to enhance the user's image prompt to create the best DALL-E image possible. Return only the enhanced prompt without any explanations or additional text.",
                input=f"Enhance this image prompt for DALL-E: {prompt}",
                temperature=self.temperature,
                max_output_tokens=500
            )
            
            # Get the refined prompt
            refined_prompt = refine_response.output
            logger.info(f"Original prompt: {prompt}")
            logger.info(f"Refined prompt: {refined_prompt}")
            
            # Use the refined prompt for DALL-E generation
            generate_prompt = refined_prompt
                
            # Generate the image with DALL-E
            response = await self.client.images.generate(
                model="dall-e-3",
                prompt=generate_prompt,
                n=1,
                size="1024x1024",
                quality="standard",
            )
            image_url = response.data[0].url
            return image_url
        except Exception as e:
            logger.error(f"Error in draw function: {e}")
            # Fall back to original prompt in case of error
            try:
                response = await self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                    quality="standard",
                )
                image_url = response.data[0].url
                return image_url
            except Exception as inner_e:
                logger.error(f"Error in fallback image generation: {inner_e}")
                raise inner_e

    async def send_start_prompt(self):
        """
        Sends the starting prompt to the chatbot model and handles the response.
        Only sends on initial startup, not on reconnects.
        """
        if self.system_prompt_sent:
            logger.info("System prompt already sent, skipping...")
            return
            
        discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
        try:
            if self.starting_prompt:
                if (discord_channel_id):
                    channel = self.get_channel(int(discord_channel_id))
                    logger.info(f"Send system prompt with size {len(self.starting_prompt)}")
                    
                    # Add a startup message to conversation history to provide context for the response
                    async with self.history_lock:
                        self.conversation_history.append({
                            "role": "user",
                            "content": "Hello! I'm starting up the bot. Please introduce yourself and let me know you're ready to help."
                        })
                    
                    response = await self.handle_response()
                    if response and len(response) > 0:
                        await channel.send(response)
                    logger.info(f"System prompt response:{response}")
                    self.system_prompt_sent = True  # Mark as sent
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
        completion = await self.client.responses.create(
            model=self.openAI_gpt_engine,
            instructions=self.starting_prompt,
            input=self._format_history_for_responses(),
            temperature=self.temperature,
            max_output_tokens=1000
        )

        response = completion.output

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

    async def render_latex(self, latex_text):
        """Generate an image from LaTeX content"""
        # Use CodeCogs API to render the LaTeX as an image
        latex_pattern = r'(\\\[|\\\(|\\begin\{.*?\}|\\frac|\\text).*?(\\\]|\\\)|\\end\{.*?\}|$)'
        found_expressions = []
        
        # Extract all LaTeX expressions
        all_matches = re.finditer(latex_pattern, latex_text, re.DOTALL)
        for match in all_matches:
            found_expressions.append(match.group(0))
            
        if not found_expressions:
            return None
            
        image_urls = []
        for expr in found_expressions:
            # Clean up the LaTeX
            clean_expr = expr.replace('\\[', '').replace('\\]', '')
            # URL encode the LaTeX formula
            encoded_expr = urllib.parse.quote(clean_expr)
            # Create URL to the rendered image
            image_url = f"https://latex.codecogs.com/png.latex?{encoded_expr}"
            image_urls.append(image_url)
            
        return image_urls
