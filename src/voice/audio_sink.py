"""
Audio Sink for recording voice from Discord
"""

import discord
import asyncio
import io
from pathlib import Path
from utils.log import logger


class AudioSink(discord.sinks.Sink):
    """Custom audio sink that records user audio"""

    def __init__(self):
        super().__init__()
        self.audio_data = {}  # user_id -> audio bytes
        self.recording = True

    def write(self, data, user):
        """Called when audio data is received from a user"""
        if user and self.recording:
            if user not in self.audio_data:
                self.audio_data[user] = bytearray()
            self.audio_data[user].extend(data.file.read())

    def cleanup(self):
        """Called when recording stops"""
        self.recording = False

    def get_audio(self, user):
        """Get recorded audio for a specific user"""
        return bytes(self.audio_data.get(user, bytearray()))


class VoiceListener:
    """Handles listening to voice channel and transcribing audio"""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.audio_cache_dir = Path("audio_cache")
        self.audio_cache_dir.mkdir(exist_ok=True)

    async def listen_for_speech(self, voice_client: discord.VoiceClient, timeout: int = 30) -> tuple[str, int]:
        """
        Listen for speech in voice channel and return transcription

        Args:
            voice_client: Connected voice client
            timeout: Maximum seconds to listen

        Returns:
            Tuple of (transcribed_text, user_id) or (None, None) if timeout
        """
        logger.info(f"Starting to listen for speech (timeout: {timeout}s)")

        # Create audio sink
        sink = AudioSink()

        # Start recording
        voice_client.start_recording(
            sink,
            lambda *args: logger.info("Recording stopped"),
            lambda *args: logger.error(f"Recording error: {args}")
        )

        # Wait for timeout or audio
        await asyncio.sleep(timeout)

        # Stop recording
        voice_client.stop_recording()
        logger.info("Stopped recording")

        # Find user with most audio data
        if not sink.audio_data:
            logger.warning("No audio data recorded")
            return None, None

        # Get user who spoke the most
        user_id = max(sink.audio_data.keys(), key=lambda u: len(sink.audio_data[u]))
        audio_bytes = sink.get_audio(user_id)

        if len(audio_bytes) < 1000:  # Minimum audio threshold
            logger.warning("Audio data too short")
            return None, None

        logger.info(f"Recorded {len(audio_bytes)} bytes from user {user_id}")

        # Save to file for Whisper API
        audio_file = self.audio_cache_dir / f"recording_{user_id}.wav"
        with open(audio_file, 'wb') as f:
            f.write(audio_bytes)

        # Transcribe with Whisper
        try:
            transcription = await self.transcribe_audio(audio_file)
            logger.info(f"Transcribed: {transcription}")
            return transcription, user_id
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
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
