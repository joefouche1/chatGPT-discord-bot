import os
import re
import random
import asyncio
import discord
from discord import Embed
import aiohttp
import io
from datetime import datetime, timedelta
import json
from typing import Optional, Tuple

import requests_cache

from utils.log import logger
from src.aclient import aclient
from src.commands.weather import get_weather
from src.commands.news import get_news
from src.commands.sports import get_sports_score, format_sports_response  # Assuming you've already moved this
from src.commands.actions import ACTION_CODES, process_action_code
from src.commands.meme import Meme

import sys
import fcntl
from pathlib import Path

# Create a cached session for requests
session = requests_cache.CachedSession('hogcache', expire_after=360)

# Instantiate the main client object
client = aclient()

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
        logger.info(
            f'{client.user} is now running! listening on {client.replying_all_discord_channel_ids}')

    @client.tree.command(name="reset", description="Complete reset conversation history")
    async def reset(interaction: discord.Interaction):
        """Command to reset the conversation history."""
        await interaction.response.defer(ephemeral=False)
        await client.ainit_history()
        await interaction.followup.send("> **INFO: I have forgotten everything.**")
        logger.warning(
            f"\x1b[31m{client.chat_model} bot has been successfully reset\x1b[0m")

    @client.tree.command(name="help", description="Show help for the bot")
    async def help(interaction: discord.Interaction):
        """Command to show help for the bot."""
        await interaction.response.defer(ephemeral=False)
        await interaction.followup.send(""":star: **BASIC COMMANDS** \n
        - `/chat [message]` Chat with ChatGPT!


        - `/private` ChatGPT switch to private mode
        - `/public` ChatGPT switch to public mode
        - `/replyall` ChatGPT switch between replyAll mode and default mode
        - `/reset` Clear ChatGPT conversation history""")

        logger.info(
            "\x1b[31mSomeone needs help!\x1b[0m")

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

            summary = await client.get_chat_response(prompt)
            
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

    try:
        client.tree.add_command(Meme(client))
        
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
