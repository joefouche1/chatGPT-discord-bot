import os
import discord
import asyncio

from src.log import logger

from utils.message_utils import send_split_message

from dotenv import load_dotenv
from discord import app_commands

from revChatGPT.V3 import Chatbot

from asgiref.sync import sync_to_async

load_dotenv()


class aclient(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.current_channel = None
        self.activity = discord.Activity(type=discord.ActivityType.listening, name="/chat | /help")
        self.isPrivate = False
        self.is_replying_all = os.getenv("REPLYING_ALL")
        
        self.replying_all_discord_channel_ids = set(int(id) for id in os.getenv("REPLYING_ALL_DISCORD_CHANNEL_IDS").split(','))
        self.openAI_email = os.getenv("OPENAI_EMAIL")
        self.openAI_password = os.getenv("OPENAI_PASSWORD")
        self.openAI_API_key = os.getenv("OPENAI_API_KEY")
        self.openAI_gpt_engine = os.getenv("GPT_ENGINE")
        self.chatgpt_session_token = os.getenv("SESSION_TOKEN")
        self.chatgpt_access_token = os.getenv("ACCESS_TOKEN")
        self.chatgpt_paid = os.getenv("PUID")

        config_dir = os.path.abspath(f"{__file__}/../../")
        prompt_name = 'system_prompt.txt'
        prompt_path = os.path.join(config_dir, prompt_name)
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.starting_prompt = f.read()

        self.chat_model = os.getenv("CHAT_MODEL")
        self.chatbot = self.get_chatbot_model()
        self.message_queue = asyncio.Queue()

    async def handle_response(self, message: discord.Message) -> str:
        try: 
            return await asyncio.wait_for(sync_to_async(self.chatbot.ask)(message), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("chatbot.ask took over 30 seconds with no reply.")
        return "Pig brain is not responding - eat more bacon and try later."

    def get_chatbot_model(self, prompt=None) -> Chatbot:
        if not prompt:
            prompt = self.starting_prompt
        return Chatbot(api_key=self.openAI_API_key, engine=self.openAI_gpt_engine, system_prompt=prompt)

    async def process_messages(self):
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
        await message.response.defer(ephemeral=self.isPrivate) if self.is_replying_all == "False" else None
        await self.message_queue.put((message, user_message))

    async def send_message(self, message: discord.Message, user_message):
        author = message.author.id
        try:
            normal_reply = ("witty reaction" not in user_message)
            if normal_reply:
                response = (f'> **{user_message}** - <@{str(author)}> \n\n')
            else:
                response = ""

            model_out = await self.handle_response(user_message)
            response = f"{response}{model_out}"
            
            logger.info(f"Model output [{len(model_out)}]: {model_out[:100]}...")

            if normal_reply:
                await send_split_message(self, response, message)
            else:
                await message.reply(model_out, mention_author=False)
                
        except Exception as e:
            logger.exception(f"Error while sending : {e}")
            await message.channel.send(f"> **ERROR: Something went wrong, please try again later!** \n ```ERROR MESSAGE: {e}```")

    async def send_start_prompt(self):
        discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
        try:
            if self.starting_prompt:
                if (discord_channel_id):
                    channel = self.get_channel(int(discord_channel_id))
                    logger.info(f"Send system prompt with size {len(self.starting_prompt)}")
                    response = f"{await self.handle_response(self.starting_prompt)}"
                    await channel.send(response)
                    logger.info(f"System prompt response:{response}")
                else:
                    logger.info("No Channel selected. Skip sending system prompt.")
            else:
                logger.info("Not given starting prompt. Skipping...")
        except Exception as e:
            logger.exception(f"Error while sending system prompt: {e}")
