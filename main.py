from src import bot
from utils.log import logger
from dotenv import load_dotenv
import discord

if __name__ == '__main__':
    # Load opus library for voice support
    if not discord.opus.is_loaded():
        try:
            discord.opus.load_opus('libopus.so.0')
            logger.info("Opus library loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load opus library: {e}")
            logger.warning("Voice features may not work properly")

    bot.run_discord_bot()
