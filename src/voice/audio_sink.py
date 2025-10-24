"""
Audio Sink for recording voice from Discord
"""

import discord
import asyncio
import io
import wave
from pathlib import Path
from utils.log import logger


class AudioBuffer:
    """Buffer for collecting audio data from a user"""

    def __init__(self, user_id):
        self.user_id = user_id
        self.buffer = bytearray()

    def write(self, data):
        """Add audio data to buffer"""
        self.buffer.extend(data)

    def get_bytes(self):
        """Get buffered audio as bytes"""
        return bytes(self.buffer)

    def save_to_wav(self, filepath: Path, sample_rate=48000, channels=2):
        """Save buffer to WAV file"""
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(self.buffer)


class VoiceListener:
    """Handles listening to voice channel and transcribing audio"""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.audio_cache_dir = Path("audio_cache")
        self.audio_cache_dir.mkdir(exist_ok=True)

    async def listen_for_speech(self, voice_client: discord.VoiceClient, timeout: int = 30) -> tuple[str, int]:
        """
        Listen for speech in voice channel and return transcription

        Note: Official discord.py does not support voice recording.
        This is a placeholder that returns None.
        To enable voice recording, switch to py-cord instead of discord.py.

        Args:
            voice_client: Connected voice client
            timeout: Maximum seconds to listen

        Returns:
            Tuple of (transcribed_text, user_id) or (None, None) if timeout
        """
        logger.warning("Voice recording is not supported in official discord.py")
        logger.warning("To enable recording, install py-cord: pip install py-cord")
        logger.info(f"Would listen for speech (timeout: {timeout}s) if py-cord was installed")

        # Official discord.py doesn't support voice recording
        # You need to use py-cord (discord.py fork) for this functionality
        return None, None

    async def transcribe_audio(self, audio_file: Path) -> str:
        """
        Transcribe audio file using OpenAI Whisper API

        Args:
            audio_file: Path to audio file

        Returns:
            Transcribed text
        """
        logger.info(f"Transcribing audio file: {audio_file}")

        with open(audio_file, 'rb') as f:
            transcription = await self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"  # Optional: can auto-detect
            )

        return transcription.text

    def contains_wake_word(self, text: str, wake_words: list[str] = None) -> bool:
        """
        Check if transcribed text contains wake words

        Args:
            text: Transcribed text
            wake_words: List of wake words (default: ["pig", "hog"])

        Returns:
            True if any wake word is found
        """
        if wake_words is None:
            wake_words = ["pig", "hog"]

        text_lower = text.lower()
        for word in wake_words:
            if word.lower() in text_lower:
                logger.info(f"Found wake word '{word}' in: {text}")
                return True

        logger.info(f"No wake words found in: {text}")
        return False
