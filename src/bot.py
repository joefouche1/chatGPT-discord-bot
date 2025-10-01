import os
import re
import random
import asyncio
import discord
from discord import Embed, app_commands
import aiohttp
import io
from datetime import datetime, timedelta
import json
from typing import Optional, Tuple
import time

import requests_cache

from utils.log import logger
from src.aclient import aclient
from src.commands.weather import get_weather
from src.commands.news import get_news
from src.commands.sports import get_sports_score, format_sports_response  # Assuming you've already moved this
from src.commands.actions import ACTION_CODES, process_action_code
from src.commands.meme import Meme
from src.commands.context import ContextCommands
from src.commands.game_notifications import GameNotificationManager
from src.voice.voice_manager import VoiceManager

import sys
import fcntl
from pathlib import Path

# Create a cached session for requests
session = requests_cache.CachedSession('hogcache', expire_after=360)

# Instantiate the main client object
client = aclient()

# Initialize game notification manager
game_notifier = GameNotificationManager(client)

# Initialize voice manager
voice_manager = VoiceManager(client, client.client)

# Track processed message IDs to prevent duplicate processing
# Store message IDs with timestamps to auto-cleanup old entries
processed_messages = {}
MESSAGE_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

# Move get_weather outside of run_discord_bot
async def get_weather(channel, city: str):
    """Get weather information for a city."""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    api_key = os.getenv('WEATHER_API_KEY')
    complete_url = base_url + "appid=" + api_key + "&q=" + city
    
    try:
        response = session.get(complete_url)
        x = response.json()

        if x["cod"] != "404":
            y = x["main"]
            current_temperature = y["temp"]
            current_temperature_fahrenheit = str(
                round((current_temperature - 273.15) * 9 / 5 + 32))
            current_pressure = y["pressure"]
            current_pressure_mmHg = str(round(current_pressure * 0.750062))
            current_humidity = y["humidity"]
            z = x["weather"]
            weather_description = z[0]["description"]
            
            embed = discord.Embed(
                title=f"Weather in {city}",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Description",
                            value=f"**{weather_description}**", inline=False)
            embed.add_field(
                name="Temperature(F)", value=f"**{current_temperature_fahrenheit}°F**", inline=False)
            embed.add_field(name="Humidity(%)",
                            value=f"**{current_humidity}%**", inline=False)
            embed.add_field(name="Atmospheric Pressure(mmHg)",
                            value=f"**{current_pressure_mmHg}mmHg**", inline=False)
            
            icon = z[0]["icon"]
            embed.set_thumbnail(
                url=f"https://openweathermap.org/img/w/{icon}.png")
            
            await channel.send(embed=embed)
        else:
            await channel.send("City not found. Try format: Pittsburgh OR Pittsburgh,PA,US OR London,UK")
            
    except Exception as e:
        logger.error(f"Error getting weather: {e}")
        await channel.send("Error getting weather information. Please try again later.")

# Function to run the discord bot
def run_discord_bot():
    # Create a lock file in /tmp
    lock_file = Path("/tmp/discord_bot.lock")
    
    try:
        # Try to create and lock the file
        lock_handle = open(lock_file, 'w')
        fcntl.lockf(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # Write the PID to the lock file
        lock_handle.write(str(os.getpid()))
        lock_handle.flush()
        
        logger.info("Bot instance lock acquired")
        
    except IOError:
        logger.error("Another instance of the bot is already running")
        sys.exit(1)
        
    @client.event
    async def on_ready():
        """Event handler for when the bot is ready."""
        client.is_replying_all = os.getenv("REPLYING_ALL", "False")
        await client.tree.sync()
        await client.send_start_prompt()
        loop = asyncio.get_event_loop()
        loop.create_task(client.process_messages())
        loop.create_task(game_notifier.start_monitoring())
        logger.info(
            f'{client.user} is now running! listening on {client.replying_all_discord_channel_ids}')

    @client.tree.command(name="reset", description="Reset conversation history for this channel")
    async def reset(interaction: discord.Interaction):
        """Command to reset the conversation history for the current channel."""
        await interaction.response.defer(ephemeral=False)
        channel_id = str(interaction.channel_id)
        await client.conversation_manager.clear_context(channel_id)
        await interaction.followup.send("> **INFO: I have forgotten everything in this channel.**")
        logger.warning(
            f"\x1b[31m{client.openAI_gpt_engine} bot has been successfully reset for channel {channel_id}\x1b[0m")

    @client.tree.command(name="help", description="Show comprehensive help for the bot")
    async def help(interaction: discord.Interaction):
        """Command to show help for the bot."""
        await interaction.response.defer(ephemeral=False)

        help_embed = discord.Embed(
            title="🐷 Megapig Bot - Command Reference",
            description="Your AI-powered assistant for conversations, sports, news, images, and more!",
            color=discord.Color.blue()
        )

        # Conversation Commands
        help_embed.add_field(
            name="💬 Conversation",
            value=(
                "**@Megapig** or mention me - Chat naturally!\n"
                "**Attach images** - I can see and analyze images\n"
                "`/reset` - Clear conversation history for this channel\n"
                "`/info` - Show bot model and configuration"
            ),
            inline=False
        )

        # Sports Commands
        help_embed.add_field(
            name="🏈 Sports",
            value=(
                "`/sportsscore [query]` - Get game scores\n"
                "  *Example: `/sportsscore Boise State yesterday`*\n"
                "`/sportsnow [query]` - Get live game scores\n"
                "  *Example: `/sportsnow NFL games`*\n"
                "`/sportsnews [sport]` - Get sports news headlines\n"
                "  *Sports: nfl, nba, mlb, nhl, ncaaf, ncaab*"
            ),
            inline=False
        )

        # Game Notifications
        help_embed.add_field(
            name="🔔 Game Notifications",
            value=(
                "`/gamenotify [team] [league] [minutes]` - Subscribe to game alerts\n"
                "  *Example: `/gamenotify \"Boise State\" ncaaf 30`*\n"
                "`/gamesubscriptions` - View active subscriptions\n"
                "`/gameunsubscribe [team] [league]` - Remove subscription\n"
                "  *Leagues: ncaaf, nfl, nba, mlb, nhl, wnba, ncaab*"
            ),
            inline=False
        )

        # Image & Creative Commands
        help_embed.add_field(
            name="🎨 Images & Creative",
            value=(
                "`/draw [prompt]` - Generate AI images with DALL-E 3\n"
                "`/meme [top text] [bottom text]` - Create meme images\n"
                "`/math [latex]` - Render mathematical expressions\n"
                "  *Example: `/math x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}`*"
            ),
            inline=False
        )

        # News & Information
        help_embed.add_field(
            name="📰 News & Information",
            value=(
                "`/pignews` - Random pig/bacon related news\n"
                "`/magic8ball [question]` - Ask the magic 8-ball"
            ),
            inline=False
        )

        # AI Action Codes
        help_embed.add_field(
            name="🤖 AI Action Codes",
            value=(
                "The bot can trigger these actions when needed:\n"
                "`!SPORTS [query]` - Fetch sports scores\n"
                "`!WEATHER [city]` - Get weather information\n"
                "`!NEWS [topic]` - Search news articles\n"
                "`!DRAW [prompt]` - Generate an image\n"
                "*Just ask naturally and I'll use these when appropriate!*"
            ),
            inline=False
        )

        # Special Features
        help_embed.add_field(
            name="✨ Special Features",
            value=(
                "• **Per-Channel Memory** - I remember our conversation in each channel\n"
                "• **Image Understanding** - Attach images and ask questions about them\n"
                "• **Streaming Responses** - Real-time message streaming\n"
                "• **LaTeX Rendering** - Automatic math formula rendering\n"
                "• **Context Commands** - Use `/context` for memory management"
            ),
            inline=False
        )

        # Admin Commands
        help_embed.add_field(
            name="⚙️ Admin & Setup",
            value=(
                "`/synccommands` - Force sync commands to this server (instant)\n"
                "`/testmention` - Test bot mention handling"
            ),
            inline=False
        )

        help_embed.set_footer(text="Powered by GPT-5 • Created for the Fouché family")

        await interaction.followup.send(embed=help_embed)

        logger.info(
            f"\x1b[31m{interaction.user} requested help!\x1b[0m")

    @client.tree.command(name="synccommands", description="Force sync slash commands to this server (admin)")
    async def synccommands(interaction: discord.Interaction):
        """Force sync commands to the current guild for immediate availability."""
        await interaction.response.defer(ephemeral=True)

        try:
            # Sync to current guild for immediate availability
            guild = interaction.guild
            if guild:
                client.tree.copy_global_to(guild=guild)
                await client.tree.sync(guild=guild)
                await interaction.followup.send(
                    f"✅ Commands synced to **{guild.name}**!\n"
                    "All slash commands should now be available immediately.",
                    ephemeral=True
                )
                logger.info(f"Commands synced to guild {guild.name} ({guild.id}) by {interaction.user}")
            else:
                await interaction.followup.send(
                    "❌ This command must be used in a server, not in DMs.",
                    ephemeral=True
                )
        except Exception as e:
            logger.error(f"Error syncing commands: {e}")
            await interaction.followup.send(
                f"❌ Error syncing commands: {str(e)}",
                ephemeral=True
            )

    @client.tree.command(name="gpt5settings", description="Configure GPT-5 verbosity and reasoning effort")
    @app_commands.describe(
        verbosity="Response verbosity: low, medium, or high",
        reasoning="Reasoning effort: minimal, low, medium, or high"
    )
    @app_commands.choices(verbosity=[
        app_commands.Choice(name="Low", value="low"),
        app_commands.Choice(name="Medium", value="medium"),
        app_commands.Choice(name="High", value="high"),
    ])
    @app_commands.choices(reasoning=[
        app_commands.Choice(name="Minimal", value="minimal"),
        app_commands.Choice(name="Low", value="low"),
        app_commands.Choice(name="Medium", value="medium"),
        app_commands.Choice(name="High", value="high"),
    ])
    async def gpt5settings(
        interaction: discord.Interaction,
        verbosity: app_commands.Choice[str] = None,
        reasoning: app_commands.Choice[str] = None
    ):
        """Configure GPT-5 parameters for this channel."""
        await interaction.response.defer(ephemeral=False)

        channel_id = str(interaction.channel_id)

        # If no parameters provided, show current settings
        if verbosity is None and reasoning is None:
            current = client.get_gpt5_params(channel_id)
            await interaction.followup.send(
                f"**Current GPT-5 settings for this channel:**\n"
                f"• Verbosity: `{current['verbosity']}`\n"
                f"• Reasoning effort: `{current['reasoning_effort']}`\n\n"
                f"To change: `/gpt5settings verbosity:low reasoning:minimal`"
            )
            return

        # Update settings
        verbosity_val = verbosity.value if verbosity else None
        reasoning_val = reasoning.value if reasoning else None

        client.set_gpt5_params(channel_id, verbosity_val, reasoning_val)
        current = client.get_gpt5_params(channel_id)

        await interaction.followup.send(
            f"✅ **GPT-5 settings updated for this channel:**\n"
            f"• Verbosity: `{current['verbosity']}`\n"
            f"• Reasoning effort: `{current['reasoning_effort']}`"
        )
        logger.info(f"{interaction.user} updated GPT-5 settings in channel {channel_id}: {current}")

    @client.tree.command(name="speak", description="Join voice channel and speak a response")
    @app_commands.describe(
        query="What should I talk about?"
    )
    async def speak(interaction: discord.Interaction, *, query: str):
        """Join voice channel and speak a GPT-5 generated response"""
        await interaction.response.defer(ephemeral=False)

        # Check if user is in a voice channel
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.followup.send("❌ You need to be in a voice channel to use this command!")
            return

        voice_channel = interaction.user.voice.channel
        guild_id = interaction.guild_id

        try:
            # Join the voice channel
            await interaction.followup.send(f"🎤 Joining {voice_channel.name}...")
            await voice_manager.join_channel(voice_channel)

            # Generate response using GPT-5
            await interaction.channel.send(f"💭 Thinking about: *{query}*...")
            channel_id = str(interaction.channel_id)
            response = await client.get_chat_response(query, channel_id)

            # Speak the response
            await interaction.channel.send(f"🗣️ Speaking response...")
            await voice_manager.speak_text(guild_id, response)

            # Show the text response as well
            await interaction.channel.send(f"**Response:**\n{response}")

            # Leave after speaking
            await voice_manager.leave_channel(guild_id)
            await interaction.channel.send("👋 Left voice channel")

            logger.info(f"{interaction.user} used /speak in {voice_channel.name}: {query}")

        except Exception as e:
            logger.error(f"Error in /speak command: {e}")
            await interaction.channel.send(f"❌ Error: {str(e)}")
            # Try to leave voice channel on error
            try:
                await voice_manager.leave_channel(guild_id)
            except:
                pass

    @client.tree.command(name="info", description="Bot information")
    async def info(interaction: discord.Interaction):
        """Command to show bot information."""
        await interaction.response.defer(ephemeral=False)
        chat_engine_status = client.openAI_gpt_engine
        chat_model_status = "OpenAI API(OFFICIAL)"

        await interaction.followup.send(f"""
```fix
chat-model: {chat_model_status}
gpt-engine: {chat_engine_status}
```
""")

    @client.tree.command(name="pignews", description="Get a random news article about pigs")
    async def pignews(interaction: discord.Interaction):
        """Command to get a random news article about pigs."""
        api_key = os.getenv('NEWS_API_KEY')  # Replace with your actual API key
        url = (
            "https://newsapi.org/v2/everything?"
            "q=pig%20OR%20bacon%20OR%20hog&"
            "sortBy=relevancy&"
            "searchIn=title,description&"
            f"apiKey={api_key}"
        )
        logger.info(f"headlines: {url}")
        try:
            response = session.get(url)
        except Exception as e:
            logger.error(f"Error while retrieving news: {e}")
            await interaction.response.send_message("Error while retrieving news. Please try again later.")
            return
        data = response.json()
        if data["articles"]:
            art = None
            for _ in range(10):
                art = random.choice(data["articles"])
                adl = art['description'].lower()
                if 'pig' in adl or 'hog' in adl or 'bacon' in adl:
                    break
            else:
                art = random.choice(data["articles"])
            try:
                embed = discord.Embed(title=art['title'])
                embed.add_field(name='Description',
                                value=art['description'], inline=False)
                embed.add_field(name='Link', value=art['url'], inline=False)
                embed.set_thumbnail(url=art['urlToImage'])
                await interaction.response.send_message(embed=embed)
            except KeyError:
                await interaction.response.send_message("Data incomplete.")
        else:
            await interaction.response.send_message("No articles found.")

    @client.tree.command(name="draw", description="Generate an image with the Dall-e-3 model")
    async def draw(interaction: discord.Interaction, *, prompt: str):
        if interaction.user == client.user:
            return

        username = str(interaction.user)
        channel = str(interaction.channel)
        logger.info(
            f"\x1b[31m{username}\x1b[0m : /draw [{prompt}] in ({channel})")

        await interaction.response.defer(thinking=True, ephemeral=False)
        try:
            image_url = await client.draw(prompt)

            # Download the image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    image_data = await resp.read()

            # Create a discord.File object
            image_file = discord.File(io.BytesIO(
                image_data), filename="image.png")

            # Create an Embed object that includes the image
            embed = Embed(description=prompt, title="AI hog generated image")
            embed.set_image(url="attachment://image.png")

            # Send the embed and the image as an attachment
            await interaction.followup.send("Here you go!", embed=embed, file=image_file)

        except Exception as e:
            error_msg = str(e)
            if 'content_policy_violation' in error_msg:
                # Create a funny embed for the violation
                embed = discord.Embed(
                    title="🚫 Oink Oink! Content Policy Violation",
                    description=f"Your prompt: `{prompt}`",
                    color=discord.Color.red()
                )
                embed.add_field(
                    name="Description",
                    value="The AI pig has detected some questionable content! Let's keep things family-friendly.",
                    inline=False
                )
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send(f'> **Something Went Wrong: {e}**')

            logger.info(f"\x1b[31m{username}\x1b[0m :{e}")

    @client.tree.command(name="magic8ball", description="Ask the magic 8-ball a question")
    async def magic8ball(interaction: discord.Interaction, *, question: str):
        """Command to ask the magic 8-ball a question."""
        responses = [
            "It is certain.",
            "It is decidedly so.",
            "Without a doubt.",
            "Yes - definitely.",
            "You may rely on it.",
            "As I see it, yes.",
            "Most likely.",
            "Outlook good.",
            "Yes.",
            "Signs point to yes.",
            "Reply hazy, try again.",
            "Ask again later.",
            "Better not tell you now.",
            "Cannot predict now.",
            "Concentrate and ask again.",
            "Don't count on it.",
            "My reply is no.",
            "My sources say no.",
            "Outlook not so good.",
            "Very doubtful.",
        ]
        response = random.choice(responses)
        await interaction.response.send_message(f"Question: {question}\nAnswer: {response}")

    @client.tree.command(name="sportsnews", description="Get a funny summary of sports headlines")
    async def sportsnews(interaction: discord.Interaction, sport: str):
        """Command to get a funny summary of sports headlines for a specific sport."""
        # Map of supported sports to their ESPN API endpoints
        sport_endpoints = {
            "nfl": "football/nfl",
            "nba": "basketball/nba",
            "mlb": "baseball/mlb",
            "nhl": "hockey/nhl",
            "soccer": "soccer/eng.1",  # Using EPL as default soccer league
            "ncaaf": "football/college-football",
            "ncaab": "basketball/mens-college-basketball",
        }

        if sport.lower() not in sport_endpoints:
            await interaction.response.send_message(
                f"Sport not supported. Available options: {', '.join(sport_endpoints.keys())}")
            return

        await interaction.response.defer(thinking=True)
        
        url = f"http://site.api.espn.com/apis/site/v2/sports/{sport_endpoints[sport.lower()]}/news"
        
        try:
            response = session.get(url)
            data = response.json()
            
            if not data.get("articles"):
                await interaction.followup.send("No news articles found.")
                return

            # Ask the LLM for a funny summary
            prompt = (
                "Here are some recent sports headlines. Please provide a brief, "
                "humorous summary of 2-3 of the most interesting stories in a casual, "
                "entertaining way. Keep it light and fun!\n\n"
            )
            
            # Add the first 5 headlines to the prompt
            for article in data["articles"][:5]:
                prompt += f"- {article['headline']}\n"
                if article.get('description'):
                    prompt += f"  {article['description']}\n"

            channel_id = str(interaction.channel_id)
            summary = await client.get_chat_response(prompt, channel_id)
            
            # Create and send embed
            embed = discord.Embed(
                title=f"🎯 Latest {sport.upper()} News Rundown",
                description=summary,
                color=discord.Color.blue()
            )
            
            # Add link to first article
            if data["articles"]:
                embed.add_field(
                    name="Read More",
                    value=f"[Full Story]({data['articles'][0]['links']['web']['href']})",
                    inline=False
                )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Error fetching sports news: {e}")
            await interaction.followup.send("Error fetching sports news. Please try again later.")

    @client.tree.command(name="sportsscore", description="Get sports scores")
    async def sportsscore(interaction: discord.Interaction, query: str):
        await interaction.response.defer()  # Defer first since this might take a while

        try:
            result = await get_sports_score(query, client.client)
            embed = await format_sports_response(
                result, 
                user_name=interaction.user.name, 
                query=query
            )
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error getting sports score: {e}")
            await interaction.followup.send(f"Sorry, I couldn't find that score. Error: {str(e)}")

    @client.tree.command(name="sportsnow", description="Get live sports scores")
    async def sportsnow(interaction: discord.Interaction, query: str):
        await interaction.response.defer()  # Defer first since this might take a while

        try:
            result = await get_sports_score(query, client.client, live=True)  # Note the live=True parameter
            embed = await format_sports_response(
                result, 
                user_name=interaction.user.name, 
                query=query
            )
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error getting live sports score: {e}")
            await interaction.followup.send(f"Sorry, I couldn't find that live score. Error: {str(e)}")

    @client.event
    async def on_message(message):
        # Ignore our own messages
        if message.author == client.user:
            return
        
        # Check if we've already processed this message to prevent duplicates
        current_time = time.time()
        if message.id in processed_messages:
            logger.info(f"Skipping already processed message ID: {message.id}")
            return
            
        # Cleanup old processed messages to prevent memory leaks
        expired_messages = [msg_id for msg_id, timestamp in processed_messages.items() 
                          if current_time - timestamp > MESSAGE_EXPIRY]
        for msg_id in expired_messages:
            del processed_messages[msg_id]
        
        # Get bot's roles in this specific guild
        bot_member = message.guild.get_member(client.user.id) if message.guild else None
        bot_roles = bot_member.roles if bot_member else []
        
        # Multiple ways to check for mentions
        is_mentioned = any([
            client.user.mentioned_in(message),  # Standard mention check
            f'<@{client.user.id}>' in message.content,  # Raw mention format
            f'<@!{client.user.id}>' in message.content,  # Nickname mention format
            message.reference and message.reference.resolved and message.reference.resolved.author == client.user,  # Reply to bot
            any(role.id in [r.id for r in bot_roles] for role in message.role_mentions)  # Role mention check
        ])
        
        # Log mention detection details for debugging
        if any([client.user.mentioned_in(message), client.user.id in map(lambda x: x.id, message.mentions)]):
            logger.info(f"Mention detected in message: {message.content}")
            logger.info(f"Message mentions: {[m.id for m in message.mentions]}")
            logger.info(f"Role mentions: {[r.id for r in message.role_mentions]}")
            logger.info(f"Bot ID: {client.user.id}")
            logger.info(f"Bot roles in this guild: {[r.id for r in bot_roles]}")
        
        # Check channel conditions
        in_channels = message.channel.id in client.replying_all_discord_channel_ids
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        try:
            # Remove the mention from the message if present
            user_message = str(message.content)
            if is_mentioned:
                # Remove both <@ID> and <@!ID> mention formats
                user_message = re.sub(f'<@!?{client.user.id}>', '', user_message).strip()
                logger.info(f"After mention removal: '{user_message}'")
            
            # Set current channel for response handling
            client.current_channel = message.channel
            
            # Determine if we should respond based on three cases:
            # 1. Direct mention in any channel
            # 2. Message in allowed channel that matches regex
            # 3. Direct message
            should_respond = False
            
            if is_mentioned:
                # Always respond to mentions
                should_respond = True
            elif is_dm:
                # Always respond to DMs
                should_respond = True
            elif in_channels:
                # In allowed channels, only respond if message matches regex
                regex = os.getenv("MESSAGE_REGEX")
                should_respond = re.match(regex, user_message.lower()) is not None
            
            if should_respond:
                logger.info(f"\x1b[31m{message.author}\x1b[0m : '{user_message}' ({message.channel})")
                # Mark this message as processed with current timestamp
                processed_messages[message.id] = current_time
                
                # Add pig emoji reaction to show we're processing
                try:
                    await message.add_reaction("🐷")
                except discord.errors.Forbidden:
                    logger.warning(f"Unable to add reaction to message {message.id} - missing permissions")
                except Exception as e:
                    logger.warning(f"Failed to add pig reaction: {e}")
                
                await client.enqueue_message(message, user_message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    @client.tree.command(name="testmention", description="Test mention handling")
    async def testmention(interaction: discord.Interaction):
        """Test the bot's mention handling."""
        await interaction.response.send_message(
            f"My ID is {client.user.id}\n"
            f"Mention me like this: <@{client.user.id}>"
        )
        
    @client.tree.command(name="math", description="Render a LaTeX math expression")
    async def math(interaction: discord.Interaction, *, latex: str):
        """Render a LaTeX math expression"""
        await interaction.response.defer(thinking=True, ephemeral=False)
        
        try:
            # Clean up the LaTeX formula
            latex = latex.strip()
            
            # If not already wrapped in delimiters, add them
            if not (latex.startswith("\\[") or latex.startswith("\\begin")):
                latex = "\\[" + latex + "\\]"
                
            # Use the render_latex method to generate the image URL
            image_urls = await client.render_latex(latex)
            
            if not image_urls:
                await interaction.followup.send("Could not parse the LaTeX expression.")
                return
                
            # Create an embed for the response
            embed = discord.Embed(
                title="Math Rendering",
                description=f"```latex\n{latex}\n```",
                color=discord.Color.blue()
            )
            
            # Download and attach the first image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_urls[0]) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
                        # Create a discord.File object
                        image_file = discord.File(io.BytesIO(image_data), filename="math.png")
                        # Set the image in the embed
                        embed.set_image(url="attachment://math.png")
                        # Send the embed with the image
                        await interaction.followup.send(embed=embed, file=image_file)
                    else:
                        await interaction.followup.send(f"Failed to generate image (HTTP {resp.status})")
        
        except Exception as e:
            logger.error(f"Error rendering math: {e}")
            await interaction.followup.send(f"Error rendering math: {str(e)}")

    @client.tree.command(name="gamenotify", description="Subscribe to game start notifications for a team")
    async def gamenotify(interaction: discord.Interaction, team_name: str, league: str, lead_time: int = 30):
        """
        Subscribe to game notifications

        Args:
            team_name: Team name (e.g., "Boise State", "Ohio State", "Bengals")
            league: League code (ncaaf, nfl, nba, mlb, nhl, etc.)
            lead_time: Minutes before game to notify (default 30)
        """
        await interaction.response.defer(ephemeral=False)

        # Validate league
        valid_leagues = ["ncaaf", "nfl", "nba", "mlb", "nhl", "wnba", "ncaab"]
        if league.lower() not in valid_leagues:
            await interaction.followup.send(
                f"❌ Invalid league. Valid options: {', '.join(valid_leagues)}"
            )
            return

        # Validate lead time
        if lead_time < 5 or lead_time > 180:
            await interaction.followup.send(
                "❌ Lead time must be between 5 and 180 minutes"
            )
            return

        channel_id = str(interaction.channel_id)
        success = game_notifier.add_subscription(channel_id, team_name, league, lead_time)

        if success:
            embed = discord.Embed(
                title="✅ Game Notifications Enabled",
                description=f"You'll be notified in this channel when **{team_name}** games are starting!",
                color=discord.Color.green()
            )
            embed.add_field(name="Team", value=team_name, inline=True)
            embed.add_field(name="League", value=league.upper(), inline=True)
            embed.add_field(name="Lead Time", value=f"{lead_time} minutes", inline=True)
            embed.set_footer(text="Notifications will be checked every 15 minutes")
            await interaction.followup.send(embed=embed)
            logger.info(f"Subscription added: {team_name} ({league}) in channel {channel_id} by {interaction.user}")
        else:
            await interaction.followup.send(
                f"⚠️ You're already subscribed to **{team_name}** notifications in this channel."
            )

    @client.tree.command(name="gameunsubscribe", description="Unsubscribe from team game notifications")
    async def gameunsubscribe(interaction: discord.Interaction, team_name: str, league: str):
        """
        Unsubscribe from game notifications

        Args:
            team_name: Team name to unsubscribe from
            league: League code
        """
        await interaction.response.defer(ephemeral=False)

        channel_id = str(interaction.channel_id)
        success = game_notifier.remove_subscription(channel_id, team_name, league)

        if success:
            embed = discord.Embed(
                title="🔕 Unsubscribed",
                description=f"Game notifications for **{team_name}** ({league.upper()}) have been disabled in this channel.",
                color=discord.Color.blue()
            )
            await interaction.followup.send(embed=embed)
            logger.info(f"Subscription removed: {team_name} ({league}) from channel {channel_id} by {interaction.user}")
        else:
            await interaction.followup.send(
                f"❌ No active subscription found for **{team_name}** ({league.upper()}) in this channel."
            )

    @client.tree.command(name="gamesubscriptions", description="List all game notification subscriptions for this channel")
    async def gamesubscriptions(interaction: discord.Interaction):
        """List all active subscriptions in this channel"""
        await interaction.response.defer(ephemeral=False)

        channel_id = str(interaction.channel_id)
        subs = game_notifier.get_channel_subscriptions(channel_id)

        if not subs:
            await interaction.followup.send(
                "📭 No active game notification subscriptions in this channel.\n"
                "Use `/gamenotify` to subscribe to a team!"
            )
            return

        embed = discord.Embed(
            title="🔔 Active Game Notifications",
            description=f"Subscriptions for this channel:",
            color=discord.Color.gold()
        )

        for sub in subs:
            team = sub['team']
            league = sub['league'].upper()
            lead_time = sub.get('lead_time_minutes', 30)
            added = sub.get('added_at', 'Unknown')

            # Parse date if available
            try:
                added_date = datetime.fromisoformat(added)
                added_str = added_date.strftime("%Y-%m-%d %H:%M")
            except:
                added_str = added

            embed.add_field(
                name=f"{team} ({league})",
                value=f"⏰ {lead_time} min notice\n📅 Added: {added_str}",
                inline=False
            )

        embed.set_footer(text="Use /gameunsubscribe to remove a subscription")
        await interaction.followup.send(embed=embed)

    try:
        # Add the cogs
        asyncio.run(client.add_cog(Meme(client)))
        asyncio.run(client.add_cog(ContextCommands(client)))
        
        TOKEN = os.getenv("DISCORD_BOT_TOKEN")
        client.run(TOKEN)
    finally:
        # Clean up the lock file when the bot exits
        try:
            fcntl.lockf(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()
            lock_file.unlink()  # Delete the lock file
            logger.info("Bot instance lock released")
        except Exception as e:
            logger.error(f"Error cleaning up lock file: {e}")
