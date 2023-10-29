import os
import re
import random
import asyncio
import discord
import requests_cache

from src.log import logger
from src.aclient import aclient


# Make cached thing
session = requests_cache.CachedSession('hogcache', expire_after=360)

# Make main client object
client = aclient()


def run_discord_bot():
    @client.event
    async def on_ready():
        client.is_replying_all = os.getenv("REPLYING_ALL", "False")
        await client.send_start_prompt()
        await client.tree.sync()
        loop = asyncio.get_event_loop()
        loop.create_task(client.process_messages())
        logger.info(
            f'{client.user} is now running! listening on {client.replying_all_discord_channel_ids}')

    @client.tree.command(name="chat", description="Have a chat with ChatGPT")
    async def chat(interaction: discord.Interaction, *, message: str):
        if client.is_replying_all == "True":
            await interaction.response.defer(ephemeral=False)
            await interaction.followup.send(
                "> **WARN: You already on replyAll mode. If you want to use the Slash Command, switch to normal mode by using `/replyall` again**")
            logger.warning(
                "\x1b[31mYou already on replyAll mode, can't use slash command!\x1b[0m")
            return
        if interaction.user == client.user:
            return
        username = str(interaction.user)
        client.current_channel = interaction.channel
        logger.info(
            f"\x1b[31m{username}\x1b[0m : /chat [{message}] in ({client.current_channel})")

        await client.enqueue_message(interaction, message)

    @client.tree.command(name="private", description="Toggle private access")
    async def private(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)
        if not client.isPrivate:
            client.isPrivate = not client.isPrivate
            logger.warning("\x1b[31mSwitch to private mode\x1b[0m")
            await interaction.followup.send(
                "> **INFO: Next, the response will be sent via private reply. If you want to switch back to public mode, use `/public`**")
        else:
            logger.info("You already on private mode!")
            await interaction.followup.send(
                "> **WARN: You already on private mode. If you want to switch to public mode, use `/public`**")

    @client.tree.command(name="public", description="Toggle public access")
    async def public(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)
        if client.isPrivate:
            client.isPrivate = not client.isPrivate
            await interaction.followup.send(
                "> **INFO: Next, the response will be sent to the channel directly. If you want to switch back to private mode, use `/private`**")
            logger.warning("\x1b[31mSwitch to public mode\x1b[0m")
        else:
            await interaction.followup.send(
                "> **WARN: You already on public mode. If you want to switch to private mode, use `/private`**")
            logger.info("You already on public mode!")

    @client.tree.command(name="replyall", description="Toggle replyAll access")
    async def replyall(interaction: discord.Interaction):
        client.replying_all_discord_channel_id = str(interaction.channel_id)
        await interaction.response.defer(ephemeral=False)
        if client.is_replying_all == "True":
            client.is_replying_all = "False"
            await interaction.followup.send(
                "> **INFO: Next, the bot will response to the Slash Command. If you want to switch back to replyAll mode, use `/replyAll` again**")
            logger.warning("\x1b[31mSwitch to normal mode\x1b[0m")
        elif client.is_replying_all == "False":
            client.is_replying_all = "True"
            await interaction.followup.send(
                "> **INFO: Next, the bot will disable Slash Command and responding to all message in this channel only. If you want to switch back to normal mode, use `/replyAll` again**")
            logger.warning("\x1b[31mSwitch to replyAll mode\x1b[0m")

    @client.tree.command(name="reset", description="Complete reset conversation history")
    async def reset(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)
        client.chatbot = client.get_chatbot_model()
        await interaction.followup.send("> **INFO: I have forgotten everything.**")
        logger.warning(
            f"\x1b[31m{client.chat_model} bot has been successfully reset\x1b[0m")

    @client.tree.command(name="help", description="Show help for the bot")
    async def help(interaction: discord.Interaction):
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
        api_key = os.getenv('NEWS_API_KEY')  # Replace with your actual API key
        url = f"https://newsapi.org/v2/everything?q=pig%20OR%20bacon%20OR%20hog&sortBy=relevancy&searchIn=title,description&apiKey={api_key}"
        logger.info(f"headlines: {url}")
        response = session.get(url)
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
                embed.add_field(name='Description', value=art['description'], inline=False)
                embed.add_field(name='Link', value=art['url'], inline=False)
                embed.set_thumbnail(url=art['urlToImage'])
                await interaction.response.send_message(embed=embed)
            except KeyError:
                await interaction.response.send_message("Data incomplete.")
        else:
            await interaction.response.send_message("No articles found.")

    @client.tree.command(name='weather', description='Weather')    
    async def weather(ctx, *, city: str):
        city_name = city
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        api_key = os.getenv('WEATHER_API_KEY')
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = session.get(complete_url)
        # logger.info(f"weather {complete_url}")
        x = response.json()
 
        if x["cod"] != "404":
            y = x["main"]
            current_temperature = y["temp"]
            current_temperature_fahrenheit = str(round((current_temperature - 273.15) * 9 / 5 + 32))
            current_pressure = y["pressure"]
            current_pressure_mmHg = str(round(current_pressure * 0.750062))  # Convert pressure from hPa to mmHg
            current_humidity = y["humidity"]
            z = x["weather"]
            weather_description = z[0]["description"]
            embed = discord.Embed(title=f"Weather in {city_name}", color=ctx.guild.me.top_role.color, timestamp=ctx.created_at)
            embed.add_field(name="Description", value=f"**{weather_description}**", inline=False)
            embed.add_field(name="Temperature(F)", value=f"**{current_temperature_fahrenheit}°F**", inline=False)
            embed.add_field(name="Humidity(%)", value=f"**{current_humidity}%**", inline=False)
            embed.add_field(name="Atmospheric Pressure(mmHg)", value=f"**{current_pressure_mmHg}mmHg**", inline=False)
            icon = z[0]["icon"]
            embed.set_thumbnail(url=f"https://openweathermap.org/img/w/{icon}.png")
            embed.set_footer(text=f"Requested by {ctx.user.name}")
            await ctx.response.send_message(embed=embed)
        else:
            await ctx.response.send_message("City not found. Type as follows: Pittsburgh OR Pittsburgh,PA,US OR London,UK")

    @client.tree.command(name="magic8ball", description="Ask the magic 8-ball a question")
    async def magic8ball(interaction: discord.Interaction, *, question: str):
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

    @client.event
    async def on_message(message):
        in_channels = (message.channel.id in client.replying_all_discord_channel_ids)
        is_dm = isinstance(message.channel, discord.DMChannel)

        username = str(message.author)
        user_message = str(message.content)
        writer = str(message.author.name)
       
        if (message.author != client.user) and (in_channels or is_dm):
            client.current_channel = message.channel
            regex = os.getenv("MESSAGE_REGEX")
            try:
                if re.match(regex, user_message.lower()) or client.user.mentioned_in(message) or is_dm:
                    # ACTIVATE THE PIG
                    if ',' in user_message:
                        user_message = user_message.split(',', 1)[1].strip()
                    logger.info(
                        f"\x1b[31m{username}\x1b[0m : '{user_message}' ({client.current_channel})")
                    await client.enqueue_message(message, user_message)

                    if random.random() < 0.25:
                        if "Howler" in writer:
                            await message.add_reaction("🐷")
                        elif "Joe" in writer:
                            await message.add_reaction("🧠")
            except re.error:
                logger.error(f"Invalid regex: {regex}")

        if (message.author == client.user) and message.embeds:
            ezero = message.embeds[0]
            logger.info(f"Found something we said with embeds: {ezero.fields}")
            react_to = None
            for field in ezero.fields:
                # logger.info(f"Field name: {field.name}, Field value: {field.value}")
                if field.name == 'Description':
                    react_to = ezero.title + ": " + field.value
                    logger.info(f"Found summary for {ezero.title} posting")
            if react_to:
                user_message = f"Give a short, witty reaction to this headline: {react_to}"
                logger.info(f"\x1b[31m{username}\x1b[0m : '{user_message}' ({client.current_channel})")
                await client.enqueue_message(message, user_message)
        #     response = client.handle_response(user_message)
        #     await message.channel.send(response)
        # else:
        #    logger.info(f"ignoring message in {message.channel.id} from {message.author}: set membership {in_channels}")
        # else:
        #     logger.exception(
        #         "replying_all_discord_channel_id not found, please use the command `/replyall` again.")

    TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    client.run(TOKEN)
