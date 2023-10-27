import os
import re
import random
import asyncio
import discord


from src.log import logger
from src.aclient import aclient


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
        #     response = client.handle_response(user_message)
        #     await message.channel.send(response)
        # else:
        #    logger.info(f"ignoring message in {message.channel.id} from {message.author}: set membership {in_channels}")
        # else:
        #     logger.exception(
        #         "replying_all_discord_channel_id not found, please use the command `/replyall` again.")

    TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    client.run(TOKEN)
