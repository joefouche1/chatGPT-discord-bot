import os
import discord
import asyncio
import aiohttp
import io
import re
import urllib.parse
import base64
import json
from PIL import Image

from utils.log import logger

from utils.message_utils import send_split_message
from src.conversation_manager import EnhancedConversationManager, JSONMemoryStorage
from src.memory import HybridMemoryStorage, MemoryExtractor, extract_and_store_memory

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


# ConversationManager moved to conversation_manager.py for enhanced per-channel support


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
        intents.voice_states = True  # Required for voice connections
        #intents.members = True
        super().__init__(command_prefix="!", intents=intents, **kwargs)
        
        # Initialize enhanced conversation manager with persistence support
        enable_persistence = os.getenv("ENABLE_PERSISTENCE", "false").lower() == "true"
        self.conversation_manager = EnhancedConversationManager(
            storage=JSONMemoryStorage("conversation_memories"),
            enable_persistence=enable_persistence,
            max_inactive_hours=168  # 7 days
        )

        # Initialize Phase 2 persistent memory system
        enable_memory_extraction = os.getenv("ENABLE_MEMORY_EXTRACTION", "true").lower() == "true"
        self.memory_extraction_enabled = enable_memory_extraction
        
        if enable_memory_extraction:
            try:
                # Initialize HybridMemoryStorage (will auto-detect OpenAI or sentence-transformers)
                self.memory_storage = HybridMemoryStorage(base_path="memory")
                logger.info("Initialized HybridMemoryStorage for persistent memory")
            except Exception as e:
                logger.warning(f"Failed to initialize HybridMemoryStorage, disabling memory extraction: {e}")
                self.memory_extraction_enabled = False
                self.memory_storage = None
        else:
            logger.info("Memory extraction disabled via ENABLE_MEMORY_EXTRACTION env var")
            self.memory_storage = None

        # Use the built-in command tree from commands.Bot (already initialized by super().__init__)
        self.current_channel = None
        self.activity = discord.Activity(
            type=discord.ActivityType.listening, name="bacon sizzle")

        self.replying_all_discord_channel_ids = set(
            int(id) for id in os.getenv("REPLYING_ALL_DISCORD_CHANNEL_IDS").split(','))

        self.openAI_API_key = os.getenv("OPENAI_API_KEY")
        self.openAI_gpt_engine = os.getenv("GPT_ENGINE", "gpt-5")
        self.temperature = 1.0 # expected by gpt5
        self.client = AsyncOpenAI()

        # Initialize memory extractor (Phase 2)
        if self.memory_extraction_enabled:
            self.memory_extractor = MemoryExtractor(self.client)
            logger.info("Initialized MemoryExtractor for automatic memory capture")
        else:
            self.memory_extractor = None

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

        # No longer using single conversation_history - managed per channel
        self.history_lock = asyncio.Lock()  # Keep for compatibility
        self.message_queue = asyncio.Queue()
        self.system_prompt_sent = False  # Track if system prompt has been sent

        self.rate_limiter = commands.CooldownMapping.from_cooldown(1, 5.0, commands.BucketType.user)
        
        # Image cache to store low-resolution versions of Discord images
        self.image_cache = {}  # {original_url: base64_data_url}
        self.max_cache_size = 50  # Maximum number of images to cache

        # GPT-5 settings
        self.gpt5_settings_file = "gpt5_settings.json"
        self.gpt5_settings = self.load_gpt5_settings()

    # init_history methods no longer needed - handled by conversation manager per channel

    def load_gpt5_settings(self):
        """Load GPT-5 settings from file"""
        try:
            if os.path.exists(self.gpt5_settings_file):
                with open(self.gpt5_settings_file, 'r') as f:
                    settings = json.load(f)
                logger.info(f"Loaded GPT-5 settings: {settings}")
                return settings
            else:
                # Default settings
                default_settings = {
                    "global": {
                        "verbosity": "low",
                        "reasoning_effort": "low"
                    }
                }
                self.save_gpt5_settings(default_settings)
                return default_settings
        except Exception as e:
            logger.error(f"Error loading GPT-5 settings: {e}")
            return {
                "global": {
                    "verbosity": "low",
                    "reasoning_effort": "low"
                }
            }

    def save_gpt5_settings(self, settings):
        """Save GPT-5 settings to file"""
        try:
            with open(self.gpt5_settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            logger.info("Saved GPT-5 settings to file")
        except Exception as e:
            logger.error(f"Error saving GPT-5 settings: {e}")

    def get_gpt5_params(self, channel_id: str = None):
        """Get GPT-5 parameters for a channel or use global defaults"""
        channel_key = str(channel_id) if channel_id else None

        # Check if channel has specific settings
        if channel_key and channel_key in self.gpt5_settings:
            settings = self.gpt5_settings[channel_key]
        else:
            settings = self.gpt5_settings.get("global", {
                "verbosity": "low",
                "reasoning_effort": "low"
            })

        return {
            "verbosity": settings.get("verbosity", "low"),
            "reasoning_effort": settings.get("reasoning_effort", "low")
        }

    def set_gpt5_params(self, channel_id: str, verbosity: str = None, reasoning_effort: str = None):
        """Set GPT-5 parameters for a channel"""
        channel_key = str(channel_id)

        if channel_key not in self.gpt5_settings:
            self.gpt5_settings[channel_key] = {}

        if verbosity:
            self.gpt5_settings[channel_key]["verbosity"] = verbosity
        if reasoning_effort:
            self.gpt5_settings[channel_key]["reasoning_effort"] = reasoning_effort

        self.save_gpt5_settings(self.gpt5_settings)
        logger.info(f"Updated GPT-5 settings for channel {channel_id}: {self.gpt5_settings[channel_key]}")

    async def _format_history_for_responses(self, channel_id: str):
        """
        Format conversation history for the Responses API.
        Combines all messages into a single input string.
        """
        conversation_history = await self.conversation_manager.get_history(channel_id)
        if not conversation_history:
            return ""
        
        formatted_parts = []
        for msg in conversation_history:
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

    async def _maintain_typing(self, channel):
        """
        Maintain typing indicator for as long as needed.
        Discord's typing indicator automatically stops after ~10 seconds, so we need to keep refreshing it.
        """
        try:
            while True:
                # Use typing() method if trigger_typing doesn't exist
                try:
                    if hasattr(channel, 'trigger_typing'):
                        await channel.trigger_typing()
                    elif hasattr(channel, 'typing'):
                        async with channel.typing():
                            await asyncio.sleep(8)
                    else:
                        logger.warning("Channel doesn't support typing indicators")
                        break
                except AttributeError:
                    logger.warning("Channel doesn't support typing indicators")
                    break
                # Wait 8 seconds before refreshing (typing lasts ~10 seconds)
                if hasattr(channel, 'trigger_typing'):
                    await asyncio.sleep(8)
        except asyncio.CancelledError:
            # This is expected when we cancel the task
            logger.debug("Typing maintenance cancelled")
            raise
        except Exception as e:
            logger.warning(f"Error maintaining typing indicator: {e}")

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

    async def handle_response(self, user_message: str = None, channel_id: str = None, discord_message: discord.Message = None) -> str:
        """
        Handles the response from the chatbot model and adds it to the conversation history.
        """
        response = ""
        spoke = False
        spoke2 = False
        brain_reaction_added = False
        start_time = asyncio.get_event_loop().time()
        last_token_time = start_time

        async_generator = self.ask_stream_async(user_message, channel_id)

        # Background task to add brain emoji after 30 seconds
        async def add_brain_emoji_after_delay():
            await asyncio.sleep(30)
            if not spoke and discord_message and not brain_reaction_added:
                try:
                    await discord_message.add_reaction("🧠")
                    logger.info("Added brain emoji reaction after 30 seconds of processing")
                    return True
                except Exception as e:
                    logger.warning(f"Could not add brain emoji reaction: {e}")
            return False

        brain_emoji_task = asyncio.create_task(add_brain_emoji_after_delay())

        try:
            while True:
                try:
                    # Use different timeouts depending on whether we've received tokens
                    # First token: 120 seconds (for reasoning models)
                    # Subsequent tokens: 60 seconds (stream should be flowing)
                    timeout = 120 if not spoke else 60

                    token = await asyncio.wait_for(async_generator.__anext__(), timeout=timeout)

                    if token is not None:
                        response += token
                        last_token_time = asyncio.get_event_loop().time()

                        if len(response) > 0 and not spoke:
                            logger.info(f"Response has started with: {response}")
                            spoke = True
                            # Cancel brain emoji task if we got a response
                            if not brain_emoji_task.done():
                                brain_emoji_task.cancel()
                        elif len(response) % 500 > 0 and spoke and (not spoke2):
                            logger.info(f"Response is up to {len(response)}")
                            spoke2 = True

                except asyncio.TimeoutError:
                    elapsed_time = asyncio.get_event_loop().time() - start_time

                    # Check if brain emoji was added by background task
                    if brain_emoji_task.done() and not brain_reaction_added:
                        try:
                            brain_reaction_added = await brain_emoji_task
                        except:
                            pass

                    # After 5 minutes total, give up
                    if elapsed_time >= 300:
                        logger.error(f"Response timed out after {elapsed_time:.1f} seconds")
                        # Remove brain emoji if it was added
                        if brain_reaction_added and discord_message:
                            try:
                                await discord_message.remove_reaction("🧠", self.user)
                            except Exception:
                                pass
                        return "⚠️ **Request timed out**: The model took longer than 5 minutes to respond. This might be due to high load on reasoning models. Please try again or use a simpler query."

                    # If we've been waiting too long for first token, log it
                    if not spoke:
                        logger.warning(f"Still waiting for first token after {elapsed_time:.1f} seconds")
                    else:
                        logger.warning(f"Token stream stalled after {elapsed_time:.1f} seconds")

                    # Continue waiting
                    continue

                except StopAsyncIteration:
                    break

        except Exception as e:
            logger.error(f"Unexpected error in handle_response: {e}")
            # Cancel brain emoji task
            if not brain_emoji_task.done():
                brain_emoji_task.cancel()
            # Remove brain emoji if it was added
            if brain_reaction_added and discord_message:
                try:
                    await discord_message.remove_reaction("🧠", self.user)
                except Exception:
                    pass
            raise
        finally:
            # Make sure brain emoji task is cancelled
            if not brain_emoji_task.done():
                brain_emoji_task.cancel()
                try:
                    await brain_emoji_task
                except asyncio.CancelledError:
                    pass

        # Remove brain emoji reaction if response completed successfully
        if brain_reaction_added and discord_message:
            try:
                await discord_message.remove_reaction("🧠", self.user)
            except Exception:
                pass

        # Check if brain emoji task completed
        if brain_emoji_task.done() and not brain_reaction_added:
            try:
                brain_reaction_added = await brain_emoji_task
                # Remove it if it was added
                if brain_reaction_added and discord_message:
                    try:
                        await discord_message.remove_reaction("🧠", self.user)
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        # Check if we got an empty response
        if not response or len(response.strip()) == 0:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Received empty response after {elapsed_time:.1f} seconds")
            return (
                "⚠️ **Empty Response Received**\n\n"
                "The model returned an empty response. This can happen when:\n"
                "• The model is still thinking but the stream ended prematurely\n"
                "• The API encountered an internal error\n"
                "• The response was filtered by content policies\n"
                f"• Stream ran for {elapsed_time:.1f} seconds before ending\n\n"
                "Please try:\n"
                "1. Rephrasing your question\n"
                "2. Breaking complex queries into simpler parts\n"
                "3. Waiting a moment and trying again"
            )

        # Check for LaTeX expressions and process them
        if response and "\\begin{" in response or "\\frac" in response or "\\text" in response:
            # If we find LaTeX, return a special marker to indicate it needs image processing
            if channel_id:
                await self.conversation_manager.add_message(channel_id, {
                    "role": "assistant",
                    "content": response
                })
            return f"__LATEX_RENDER__{response}"
        
        # Add the model's response to the conversation history
        if channel_id:
            await self.conversation_manager.add_message(channel_id, {
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

        # Phase 2: Extract and store memory after response completes
        if self.memory_extraction_enabled and user_message and response:
            try:
                # Get Discord metadata for richer context
                username = None
                user_id = None
                channel_name = None
                guild_name = None
                
                if discord_message:
                    user_id = str(discord_message.author.id)
                    username = str(discord_message.author.name)
                    channel_name = discord_message.channel.name if hasattr(discord_message.channel, 'name') else None
                    guild_name = discord_message.guild.name if discord_message.guild else None
                
                # Extract and store memory asynchronously (don't block response)
                asyncio.create_task(
                    extract_and_store_memory(
                        extractor=self.memory_extractor,
                        storage=self.memory_storage,
                        user_message=user_message,
                        assistant_message=response,
                        channel_id=channel_id,
                        user_id=user_id,
                        username=username,
                        channel_name=channel_name,
                        guild_name=guild_name
                    )
                )
                logger.debug(f"Triggered memory extraction for channel {channel_id}")
            except Exception as e:
                logger.error(f"Error triggering memory extraction: {e}", exc_info=True)

        return response

    async def handle_image_attachment(self, message: discord.Message, user_message: str) -> str:
        """
            Handle processing of image attachments and add them to conversation history cleanly.
        """
        original_img_url = message.attachments[0].url + "?width=500&height=500"
        channel_id = str(message.channel.id)

        logger.info(f"Prompting with photo at {original_img_url}")
        
        # Cache the image to avoid CDN expiration issues
        cached_img_url = await self.cache_discord_image(original_img_url)
        
        # Append the image + text as a single user message into the persistent history
        # Use cached image URL to prevent CDN expiration
        await self.conversation_manager.add_message(channel_id, {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": cached_img_url}}
            ]
        })

        await self.__truncate_conversation(channel_id)

        # Format multimodal input for Responses API
        # Wrap content under a single user message per current schema
        combined_text = f"{await self._format_history_for_responses(channel_id)}\n\nUser: {user_message}"
        multimodal_input = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": combined_text},
                    {"type": "input_image", "image_url": cached_img_url}
                ]
            }
        ]

        # Use GPT-5 for image understanding while preserving context
        gpt5_params = self.get_gpt5_params(channel_id)
        completion = await self.client.responses.create(
            model="gpt-5",
            instructions=self.starting_prompt,
            input=multimodal_input,
            temperature=self.temperature,
            max_output_tokens=10000,
            text={"format": {"type": "text"}, "verbosity": gpt5_params["verbosity"]},
            reasoning={"effort": gpt5_params["reasoning_effort"]}
        )

        reply = getattr(completion, "output_text", None) or getattr(completion, "output", "")

        # Add assistant's reply to history
        await self.conversation_manager.add_message(channel_id, {
            "role": "assistant",
            "content": reply
        })

        return reply

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    async def get_token_count(self, channel_id: str) -> int:
        """
        Get token count for a specific channel's conversation
        """
        try:
            # Try to get the encoding for the model
            encoding = tiktoken.encoding_for_model(self.openAI_gpt_engine)
        except KeyError:
            # If model not found, use cl100k_base as default
            logger.info(f"Model {self.openAI_gpt_engine} not found in tiktoken mappings, using cl100k_base encoding")
            encoding = tiktoken.get_encoding("cl100k_base")
            
        num_tokens = 0
        conversation_history = await self.conversation_manager.get_history(channel_id)
        for message in conversation_history:
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

    async def __truncate_conversation(self, channel_id: str) -> None:
        """
        Truncate the conversation for a specific channel
        """
        while True:
            conversation_history = await self.conversation_manager.get_history(channel_id)
            if (await self.get_token_count(channel_id) > self.truncate_limit and len(conversation_history) > 1):
                # Remove the second message (keep system/first message)
                context = await self.conversation_manager.get_or_create_context(channel_id)
                async with context.lock:
                    if len(context.conversation_history) > 1:
                        context.conversation_history.pop(1)
                        logger.info(f"popped an old message from channel {channel_id}!")
            else:
                break

    async def ask_stream_async(self, message=None, channel_id=None):
        """
        Adds the user's message to the conversation history and initializes
        the chat models with the conversation history.
        """
        engine = self.openAI_gpt_engine
        
        if not channel_id:
            logger.warning("No channel_id provided to ask_stream_async")
            return
            
        if message:
            # Add the user's message to the conversation history
            await self.conversation_manager.add_message(channel_id, {
                "role": "user",
                "content": message
            })

        await self.__truncate_conversation(channel_id)

        formatted_input = await self._format_history_for_responses(channel_id)
        logger.info(f"Sending to API - Input length: {len(formatted_input)}")
        logger.info(f"Input preview: {formatted_input[:200]}...")

        # Initialize the chat models with the conversation history
        # Use higher token limit for GPT-5 to accommodate reasoning tokens
        gpt5_params = self.get_gpt5_params(channel_id)
        stream = await self.client.responses.create(
            model=engine,
            instructions=self.starting_prompt,
            input=formatted_input,
            temperature=self.temperature,
            max_output_tokens=10000,  # Increased from 4000 to allow for reasoning + output
            text={"format": {"type": "text"}, "verbosity": gpt5_params["verbosity"]},
            reasoning={"effort": gpt5_params["reasoning_effort"]},
            stream=True
        )

        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type in ("response.output_text.delta", "content.text.delta"):
                text_delta = getattr(event, "delta", None)
                if not text_delta:
                    text_delta = getattr(event, "text", "")
                if text_delta:
                    yield text_delta
            elif event_type in ("response.output_text.done", "content.text.done", "response.completed"):
                continue

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
        channel_id = str(message.channel.id)
        
        # Get or create context for this channel
        channel_name = getattr(message.channel, 'name', None)
        guild_name = getattr(message.guild, 'name', None) if message.guild else None
        await self.conversation_manager.get_or_create_context(channel_id, channel_name, guild_name)
        
        # Start typing indicator
        typing_task = None
        try:
            # Start typing indicator (it will automatically stop after ~10 seconds, so we need to keep it alive)
            typing_task = asyncio.create_task(self._maintain_typing(response_channel))
        except Exception as e:
            logger.warning(f"Failed to start typing indicator: {e}")
        
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

                model_out = await self.handle_response(user_message=user_message, channel_id=channel_id, discord_message=message)
                
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
        finally:
            # Ensure typing indicator is stopped
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
                logger.debug("Typing indicator stopped")

    async def draw(self, prompt) -> list[str]:
        """Generate image using DALL-E model"""
        try:
            # Use GPT-5 to refine/improve the prompt
            gpt5_params = self.get_gpt5_params()  # Use global settings for draw function
            refine_response = await self.client.responses.create(
                model="gpt-5",
                instructions="You are a helpful image prompt engineer. Your task is to enhance the user's image prompt to create the best DALL-E image possible. Return only the enhanced prompt without any explanations or additional text.",
                input=f"Enhance this image prompt for DALL-E: {prompt}",
                temperature=self.temperature,
                max_output_tokens=1000,
                text={"format": {"type": "text"}, "verbosity": "low"},
                reasoning={"effort": "low"}
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
                    channel_id = str(discord_channel_id)
                    logger.info(f"Send system prompt with size {len(self.starting_prompt)}")
                    
                    # Get or create context for startup channel
                    channel_name = getattr(channel, 'name', None)
                    guild_name = getattr(channel.guild, 'name', None) if channel.guild else None
                    await self.conversation_manager.get_or_create_context(channel_id, channel_name, guild_name)
                    
                    # Add a startup message to conversation history to provide context for the response
                    await self.conversation_manager.add_message(channel_id, {
                        "role": "user",
                        "content": "Hello! I'm starting up the bot. Please introduce yourself and let me know you're ready to help."
                    })
                    
                    response = await self.handle_response(user_message=None, channel_id=channel_id, discord_message=None)
                    if response and len(response) > 0:
                        await channel.send(response)
                    logger.info(f"System prompt response:{response}")

                    # Clear the startup conversation from history so it doesn't pollute future conversations
                    await self.conversation_manager.clear_context(channel_id)
                    logger.info(f"Cleared startup greeting from conversation history for channel {channel_id}")

                    self.system_prompt_sent = True  # Mark as sent
                else:
                    logger.info(
                        "No Channel selected. Skip sending system prompt.")
            else:
                logger.info("Not given starting prompt. Skipping...")
        except Exception as e:
            logger.exception(f"Error while sending system prompt: {e}")

    async def get_chat_response(self, message: str, channel_id: str) -> str:
        """
        Gets a simple chat response without streaming for a specific channel.
        """
        # Add the user's message to conversation history
        await self.conversation_manager.add_message(channel_id, {
            "role": "user",
            "content": message
        })

        await self.__truncate_conversation(channel_id)

        # Get completion from OpenAI
        gpt5_params = self.get_gpt5_params(channel_id)
        completion = await self.client.responses.create(
            model=self.openAI_gpt_engine,
            instructions=self.starting_prompt,
            input=await self._format_history_for_responses(channel_id),
            temperature=self.temperature,
            max_output_tokens=8000,  # Increased from 1000 to allow for reasoning + output
            text={"format": {"type": "text"}, "verbosity": gpt5_params["verbosity"]},
            reasoning={"effort": gpt5_params["reasoning_effort"]}
        )

        response = getattr(completion, "output_text", None) or getattr(completion, "output", "")

        # Add the response to conversation history
        await self.conversation_manager.add_message(channel_id, {
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
