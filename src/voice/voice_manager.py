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

# Voice listening disabled - requires py-cord instead of discord.py
# from src.voice.audio_sink import VoiceListener
VoiceListener = None


class VoiceManager:
    """Manages voice connections and TTS playback"""

    def __init__(self, bot_client, openai_client: AsyncOpenAI):
        self.client = bot_client
        self.openai_client = openai_client
        self.voice_clients = {}  # guild_id -> voice_client
        self.audio_cache_dir = Path("audio_cache")
        self.audio_cache_dir.mkdir(exist_ok=True)
        self.listener = VoiceListener(openai_client) if VoiceListener else None

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
            # FFmpeg options to convert audio for Discord voice
            # Discord expects: 48kHz sample rate, 2 channels, 16-bit PCM
            ffmpeg_options = {
                'options': '-vn -ar 48000 -ac 2 -b:a 128k'
            }
            audio_source = discord.FFmpegPCMAudio(str(audio_file), **ffmpeg_options)

            # Play with error callback
            def after_callback(error):
                if error:
                    logger.error(f"Playback error: {error}")

            vc.play(audio_source, after=after_callback)
            logger.info(f"Playing audio in guild {guild_id}")

            # Wait for playback to finish
            while vc.is_playing():
                await asyncio.sleep(0.5)

            logger.info(f"Finished playing audio in guild {guild_id}")

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            raise

    async def listen_and_respond(self, guild_id: int, response_generator, timeout: int = 30) -> tuple[str, str]:
        """
        Listen for speech in voice channel and generate a response

        Note: Voice listening is currently disabled. Requires py-cord instead of discord.py.

        Args:
            guild_id: Guild ID where to listen
            response_generator: Async function that takes (question) and returns answer
            timeout: Seconds to listen for speech

        Returns:
            Tuple of (question, answer) or (None, None) if no speech detected
        """
        if not self.listener:
            logger.error("Voice listening is disabled - requires py-cord")
            raise NotImplementedError("Voice listening requires py-cord library")

        if guild_id not in self.voice_clients:
            raise ValueError(f"Not connected to voice in guild {guild_id}")

        vc = self.voice_clients[guild_id]
        if not vc.is_connected():
            raise ValueError(f"Voice client not connected in guild {guild_id}")

        # Listen for speech
        logger.info(f"Listening for speech in guild {guild_id}...")
        question, user_id = await self.listener.listen_for_speech(vc, timeout)

        if not question:
            logger.warning("No speech detected")
            return None, None

        logger.info(f"Heard from user {user_id}: {question}")

        # Check for wake words
        if not self.listener.contains_wake_word(question):
            logger.info("No wake words detected, ignoring speech")
            return None, None

        # Generate response
        logger.info("Generating response...")
        answer = await response_generator(question)

        if not answer:
            logger.error("Failed to generate response")
            return question, None

        logger.info(f"Response: {answer[:100]}...")

        # Speak the response
        await self.speak_text(guild_id, answer)

        return question, answer

    async def cleanup(self):
        """Disconnect from all voice channels"""
        for guild_id in list(self.voice_clients.keys()):
            await self.leave_channel(guild_id)
