"""
Voice Manager for Discord Bot
Handles voice channel connections and text-to-speech functionality
"""

import os
import asyncio
import discord
from pathlib import Path
from utils.log import logger
from openai import AsyncOpenAI


class VoiceManager:
    """Manages voice connections and TTS playback"""

    def __init__(self, bot_client, openai_client: AsyncOpenAI):
        self.client = bot_client
        self.openai_client = openai_client
        self.voice_clients = {}  # guild_id -> voice_client
        self.audio_cache_dir = Path("audio_cache")
        self.audio_cache_dir.mkdir(exist_ok=True)

    async def join_channel(self, voice_channel: discord.VoiceChannel) -> discord.VoiceClient:
        """Join a voice channel"""
        guild_id = voice_channel.guild.id

        # If already connected to this channel, return existing client
        if guild_id in self.voice_clients:
            vc = self.voice_clients[guild_id]
            if vc.is_connected() and vc.channel.id == voice_channel.id:
                logger.info(f"Already connected to voice channel {voice_channel.name}")
                return vc
            # If connected to different channel, move
            await vc.move_to(voice_channel)
            logger.info(f"Moved to voice channel {voice_channel.name}")
            return vc

        # Connect to the channel
        try:
            voice_client = await voice_channel.connect()
            self.voice_clients[guild_id] = voice_client
            logger.info(f"Joined voice channel {voice_channel.name} in guild {voice_channel.guild.name}")
            return voice_client
        except Exception as e:
            logger.error(f"Failed to join voice channel: {e}")
            raise

    async def leave_channel(self, guild_id: int):
        """Leave voice channel"""
        if guild_id in self.voice_clients:
            vc = self.voice_clients[guild_id]
            if vc.is_connected():
                await vc.disconnect()
                logger.info(f"Left voice channel in guild {guild_id}")
            del self.voice_clients[guild_id]

    async def generate_speech(self, text: str, voice: str = "alloy") -> Path:
        """
        Generate speech from text using OpenAI TTS

        Args:
            text: Text to convert to speech
            voice: Voice model (alloy, echo, fable, onyx, nova, shimmer)

        Returns:
            Path to the generated audio file
        """
        try:
            # Generate unique filename based on text hash
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            audio_file = self.audio_cache_dir / f"tts_{text_hash}.mp3"

            # Check if already cached
            if audio_file.exists():
                logger.info(f"Using cached audio for text: {text[:50]}...")
                return audio_file

            # Generate speech using OpenAI TTS
            logger.info(f"Generating speech for: {text[:50]}...")
            response = await self.openai_client.audio.speech.create(
                model="tts-1",  # or "tts-1-hd" for higher quality
                voice=voice,
                input=text
            )

            # Save to file
            response.stream_to_file(str(audio_file))
            logger.info(f"Generated audio file: {audio_file}")

            return audio_file

        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise

    async def speak_text(self, guild_id: int, text: str, voice: str = "alloy"):
        """
        Convert text to speech and play in voice channel

        Args:
            guild_id: Guild ID where to speak
            text: Text to speak
            voice: Voice model to use
        """
        if guild_id not in self.voice_clients:
            raise ValueError(f"Not connected to voice in guild {guild_id}")

        vc = self.voice_clients[guild_id]
        if not vc.is_connected():
            raise ValueError(f"Voice client not connected in guild {guild_id}")

        # Wait if currently playing something
        while vc.is_playing():
            await asyncio.sleep(0.5)

        # Generate audio
        audio_file = await self.generate_speech(text, voice)

        # Play audio
        try:
            audio_source = discord.FFmpegPCMAudio(str(audio_file))
            vc.play(audio_source)
            logger.info(f"Playing audio in guild {guild_id}")

            # Wait for playback to finish
            while vc.is_playing():
                await asyncio.sleep(0.5)

            logger.info(f"Finished playing audio in guild {guild_id}")

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            raise

    async def cleanup(self):
        """Disconnect from all voice channels"""
        for guild_id in list(self.voice_clients.keys()):
            await self.leave_channel(guild_id)
