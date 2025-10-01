# Voice Channel Integration Plan
## Future Enhancement for Megapig Bot

---

## Executive Summary

This document outlines a phased approach to integrate voice capabilities into the Megapig Discord bot, enabling voice channel participation for text-to-speech announcements, voice conversations, and audio-based interactions.

---

## Use Cases

### High Priority
1. **Sports Game Announcements** - Join voice channel to announce when games are starting
2. **TTS Responses** - Read out bot responses in voice channels when requested
3. **Voice Commands** - Listen to voice commands and respond via TTS
4. **Score Updates** - Periodic live score announcements during active games

### Medium Priority
5. **Music/Audio Playback** - Play sound effects or music clips
6. **Voice Conversations** - Natural voice-to-voice AI conversations
7. **Multi-User Discussions** - Participate in group voice discussions
8. **Audio Notifications** - Custom sound alerts for events

### Low Priority (Future)
9. **Voice Activity Detection** - Respond when specific keywords are spoken
10. **Audio Processing** - Analyze audio/music in voice channels

---

## Technical Architecture

### Core Components

```
┌─────────────────────────────────────────────────────┐
│                  Discord Bot Core                    │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐  ┌──────▼──────┐  ┌────▼─────┐
│ Voice Client │  │   Audio      │  │  Speech  │
│   Manager    │  │  Processing  │  │ Services │
└──────┬───────┘  └──────┬───────┘  └────┬─────┘
       │                 │                │
       │          ┌──────▼───────┐       │
       │          │  Audio Queue │       │
       │          │   & Mixing   │       │
       │          └──────────────┘       │
       │                                 │
┌──────▼─────────────────────────────────▼──────┐
│         Discord Voice Connection API           │
└────────────────────────────────────────────────┘
```

### Required Libraries

**Python Discord Voice Support:**
```python
# Primary voice support
discord.py[voice]  # Voice support for discord.py
PyNaCl            # Audio encryption/decryption

# Audio processing
pydub             # Audio manipulation
ffmpeg            # Audio codec support

# Text-to-Speech (TTS)
elevenlabs        # High-quality TTS (paid, best quality)
# OR
openai.audio.speech  # OpenAI TTS (included with GPT-5)
# OR
gTTS              # Google TTS (free, basic quality)

# Speech-to-Text (STT)
openai.audio.transcriptions  # Whisper API (best accuracy)
# OR
google-cloud-speech          # Google STT
# OR
azure-cognitiveservices-speech  # Azure STT

# Voice activity detection
webrtcvad         # Voice activity detection
```

---

## Implementation Phases

### Phase 1: Basic TTS Announcements (2-3 days)
**Goal:** Bot can join voice channels and speak announcements

**Features:**
- `/voicejoin` - Join current voice channel
- `/voiceleave` - Leave voice channel
- `/say [text]` - Speak text in voice channel
- Automatic game start announcements in voice

**Technical Tasks:**
1. Install voice dependencies (`discord.py[voice]`, `PyNaCl`, `ffmpeg`)
2. Create `VoiceManager` class to handle connections
3. Integrate OpenAI TTS or ElevenLabs
4. Add voice channel state tracking
5. Update game notification system to optionally announce in voice
6. Add queue system for multiple TTS requests

**Code Structure:**
```python
# src/voice/voice_manager.py
class VoiceManager:
    def __init__(self, bot_client):
        self.client = bot_client
        self.voice_clients = {}  # guild_id -> voice_client
        self.audio_queue = {}    # guild_id -> queue

    async def join_channel(self, voice_channel):
        """Join a voice channel"""

    async def leave_channel(self, guild_id):
        """Leave voice channel"""

    async def speak_text(self, guild_id, text):
        """Convert text to speech and play in voice"""

    async def play_audio_file(self, guild_id, file_path):
        """Play an audio file"""
```

**Estimated Complexity:** Low-Medium

---

### Phase 2: Voice Commands (STT) (3-5 days)
**Goal:** Bot can listen to voice and respond to commands

**Features:**
- `/voicelisten` - Start listening for voice commands
- Voice wake word detection ("Hey Megapig")
- Process voice commands and respond via TTS
- Voice-based sports queries
- Voice-based image generation requests

**Technical Tasks:**
1. Integrate Whisper API for speech-to-text
2. Implement audio recording and buffering
3. Add voice activity detection (VAD)
4. Create command parser for voice input
5. Implement wake word detection
6. Add conversation timeout logic
7. Handle simultaneous speakers (queue/priority)

**Code Structure:**
```python
# src/voice/voice_listener.py
class VoiceListener:
    def __init__(self, voice_manager):
        self.voice_manager = voice_manager
        self.listening = {}  # guild_id -> listening state
        self.audio_buffers = {}

    async def start_listening(self, guild_id):
        """Start recording audio from voice channel"""

    async def process_audio_chunk(self, guild_id, audio_chunk):
        """Buffer and process audio"""

    async def transcribe_audio(self, audio_data):
        """Use Whisper to transcribe"""

    async def handle_voice_command(self, guild_id, text):
        """Process voice command and respond"""
```

**Challenges:**
- Background noise filtering
- Multiple speakers talking simultaneously
- Audio quality variations
- Latency optimization

**Estimated Complexity:** Medium-High

---

### Phase 3: Natural Voice Conversations (1-2 weeks)
**Goal:** Full voice-to-voice AI conversations

**Features:**
- Continuous listening mode
- Natural conversation flow
- Voice response synthesis with emotion
- Multi-turn voice dialogues
- Per-channel voice conversation memory
- Voice profile recognition (optional)

**Technical Tasks:**
1. Implement streaming TTS for faster responses
2. Add conversation context management for voice
3. Integrate advanced voice synthesis (emotion, tone)
4. Implement interrupt handling (when user speaks)
5. Add voice personality configuration
6. Optimize latency (target <2 second response time)
7. Implement speaker diarization (who's speaking)

**Advanced Features:**
- Real-time response streaming (start speaking before full text generated)
- Voice cloning for consistent bot personality
- Background audio mixing (music + voice)
- Spatial audio positioning

**Estimated Complexity:** High

---

### Phase 4: Advanced Audio Features (2-3 weeks)
**Goal:** Rich audio experiences and multi-modal interactions

**Features:**
- Music/sound effect playback
- Audio file uploads and processing
- Live game commentary (auto-narration)
- Multi-language voice support
- Custom voice models
- Audio effects and filters
- Karaoke mode (lyrics + backing track)

**Estimated Complexity:** High

---

## Technical Considerations

### Discord Voice API Limits
- **Max voice connections:** 1 per bot per guild
- **Max audio bitrate:** 96kbps (normal), 128kbps (boosted servers)
- **Max latency target:** <300ms for good UX
- **Audio format:** Opus codec, 48kHz, 16-bit stereo

### TTS Provider Comparison

| Provider | Quality | Latency | Cost | Voices | Notes |
|----------|---------|---------|------|--------|-------|
| OpenAI TTS | Excellent | ~500ms | $15/1M chars | 6 | Included with GPT-5 |
| ElevenLabs | Best | ~800ms | $5/10K chars | 100+ | Most natural, expensive |
| Google TTS | Good | ~300ms | $4/1M chars | 220+ | Fast, reliable |
| gTTS | Basic | ~500ms | Free | 100+ | Simple, robotic |
| Azure | Excellent | ~400ms | $4/1M chars | 400+ | Neural voices |

**Recommendation:** Start with OpenAI TTS (already have API access), upgrade to ElevenLabs for production quality.

### STT Provider Comparison

| Provider | Accuracy | Latency | Cost | Languages |
|----------|----------|---------|------|-----------|
| Whisper (OpenAI) | Excellent | ~1-2s | $0.006/min | 90+ |
| Google STT | Excellent | ~1s | $0.016/min | 125+ |
| Azure STT | Excellent | ~1s | $0.016/min | 100+ |

**Recommendation:** Whisper API (best accuracy, already have access).

### Infrastructure Requirements

**Additional Resources:**
- **CPU:** +20-30% for audio processing
- **Memory:** +512MB for audio buffers
- **Storage:** Temp audio files (~100MB working space)
- **Bandwidth:** ~128kbps per voice connection

**Docker Compose Updates:**
```yaml
services:
  bot:
    # ... existing config
    volumes:
      - ./audio_cache:/app/audio_cache  # Temp audio storage
    environment:
      - ENABLE_VOICE=true
      - TTS_PROVIDER=openai
      - STT_PROVIDER=whisper
    # Install ffmpeg in container
    command: >
      sh -c "apt-get update &&
             apt-get install -y ffmpeg libopus0 &&
             python main.py"
```

---

## Command Specifications

### Phase 1 Commands

#### `/voicejoin`
**Description:** Join the voice channel you're currently in
**Parameters:** None
**Example:** `/voicejoin`

#### `/voiceleave`
**Description:** Leave the voice channel
**Parameters:** None
**Example:** `/voiceleave`

#### `/say [text]`
**Description:** Speak text in the voice channel
**Parameters:**
- `text` (required): Text to speak
**Example:** `/say Boise State scores a touchdown!`

#### `/voiceannounce [team] [league]`
**Description:** Enable voice announcements for game starts
**Parameters:**
- `team` (required): Team name
- `league` (required): League code
**Example:** `/voiceannounce "Boise State" ncaaf`

### Phase 2 Commands

#### `/voicelisten`
**Description:** Start listening for voice commands
**Parameters:** None
**Example:** `/voicelisten`

#### `/voicestop`
**Description:** Stop listening for voice commands
**Parameters:** None
**Example:** `/voicestop`

### Phase 3 Commands

#### `/voiceconverse`
**Description:** Start natural voice conversation mode
**Parameters:**
- `duration` (optional): Minutes to stay active (default: 60)
**Example:** `/voiceconverse 30`

#### `/voicepersonality [style]`
**Description:** Set voice personality style
**Parameters:**
- `style`: casual, professional, enthusiastic, sarcastic
**Example:** `/voicepersonality enthusiastic`

---

## Configuration

### Environment Variables

```bash
# Voice feature toggle
ENABLE_VOICE=true

# TTS Configuration
TTS_PROVIDER=openai  # openai, elevenlabs, google, gtts
ELEVENLABS_API_KEY=your_key_here
TTS_VOICE_ID=default  # Voice model ID

# STT Configuration
STT_PROVIDER=whisper  # whisper, google, azure
WHISPER_MODEL=whisper-1

# Voice Settings
VOICE_BITRATE=96  # kbps
VOICE_AUTO_LEAVE_TIMEOUT=300  # seconds
VOICE_MAX_RECORDING_LENGTH=30  # seconds
VOICE_WAKE_WORD=hey megapig
```

### Database Schema (Optional)

```sql
-- Voice session tracking
CREATE TABLE voice_sessions (
    id INTEGER PRIMARY KEY,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    total_speech_time INTEGER,  -- seconds
    commands_processed INTEGER
);

-- Voice preferences per guild
CREATE TABLE voice_preferences (
    guild_id TEXT PRIMARY KEY,
    tts_provider TEXT DEFAULT 'openai',
    tts_voice_id TEXT DEFAULT 'default',
    auto_join_channel_id TEXT,  -- Auto-join this channel
    announcement_enabled BOOLEAN DEFAULT false
);
```

---

## Cost Estimates

### Phase 1 (TTS Announcements)
**Monthly Usage Estimate:**
- 50 game announcements/month × 100 words = 5,000 words
- 10 manual TTS commands/day × 30 days × 50 words = 15,000 words
- **Total:** ~20,000 words/month

**Cost (OpenAI TTS):**
- 20,000 words ≈ 100,000 chars
- $15 per 1M chars = $1.50/month

### Phase 2 (Voice Commands)
**Additional Usage:**
- 100 voice commands/day × 30 days × 20 seconds = 600 minutes/month
- STT: 600 min × $0.006 = $3.60/month
- TTS responses: +30,000 words = +$4.50/month
- **Total:** ~$10/month

### Phase 3 (Natural Conversations)
**Additional Usage:**
- 30 min/day voice conversations × 30 days = 900 minutes/month
- STT: 900 min × $0.006 = $5.40/month
- TTS: ~150,000 words = $22.50/month
- **Total:** ~$38/month

**Note:** Costs scale with usage. ElevenLabs would be ~10x more expensive for TTS.

---

## Testing Strategy

### Phase 1 Testing
1. **Unit Tests:**
   - Voice client connection/disconnection
   - TTS generation and caching
   - Audio queue management

2. **Integration Tests:**
   - Join/leave voice channels
   - Play audio in voice
   - Handle concurrent TTS requests
   - Verify audio quality

3. **Manual Testing:**
   - Different voice channels
   - Various text inputs (long, short, special chars)
   - Multiple guild support
   - Network interruption recovery

### Phase 2 Testing
1. **Audio Processing Tests:**
   - Voice activity detection accuracy
   - Background noise filtering
   - Multiple speakers
   - Audio quality variations

2. **STT Accuracy Tests:**
   - Clear speech
   - Accents and dialects
   - Background noise
   - Command parsing

### Phase 3 Testing
1. **Conversation Flow Tests:**
   - Natural turn-taking
   - Interruption handling
   - Context retention
   - Latency optimization

2. **Load Tests:**
   - Multiple simultaneous conversations
   - Long-running sessions
   - Memory leak detection

---

## Security & Privacy Considerations

### Audio Privacy
- **Recording Consent:** Clearly indicate when bot is listening
- **Data Retention:** Delete audio recordings after transcription
- **Encryption:** All audio transmitted via Discord's encrypted channels
- **Opt-in:** Voice features require explicit activation

### Best Practices
1. Never store raw audio longer than necessary
2. Log only metadata (not audio content)
3. Provide clear visual/audio indicators when listening
4. Allow users to disable voice features per-guild
5. Comply with recording laws (varies by jurisdiction)

### Discord ToS Compliance
- Follow Discord's Bot Terms of Service
- Respect voice channel permissions
- Don't record without user knowledge
- Provide privacy policy if storing voice data

---

## Rollout Strategy

### Alpha Testing (Internal)
- Deploy to your Fouché family server only
- Test all Phase 1 features
- Gather feedback
- Fix bugs

### Beta Testing (Limited)
- Deploy to 2-3 trusted servers
- Monitor performance and costs
- Collect user feedback
- Refine features

### Production Release
- Deploy globally
- Monitor costs carefully
- Provide documentation
- Support user issues

---

## Alternative Approaches

### Lightweight Option: TTS Only (No STT)
**Pros:**
- Much simpler implementation
- Lower costs (~90% reduction)
- Faster to deploy
- Still covers main use case (announcements)

**Cons:**
- No voice commands
- Less interactive

### Hybrid Option: External Service
Use services like:
- **Voiceflow** - Visual voice conversation builder
- **Dasha.AI** - Voice AI platform
- **Speechly** - Voice interface platform

**Pros:**
- Pre-built infrastructure
- Managed scaling
- Better tooling

**Cons:**
- Higher costs
- Less customization
- Vendor lock-in

---

## Success Metrics

### Phase 1
- [ ] Bot can join voice channels reliably (>99% success)
- [ ] TTS audio is clear and natural
- [ ] Game announcements trigger correctly
- [ ] Latency <3 seconds from trigger to audio start
- [ ] No voice connection crashes

### Phase 2
- [ ] Voice command accuracy >90%
- [ ] Response latency <5 seconds
- [ ] Handles 3+ concurrent speakers
- [ ] Background noise rejection >80%

### Phase 3
- [ ] Conversation feels natural (user satisfaction >4/5)
- [ ] Response latency <2 seconds
- [ ] Can maintain 30+ minute conversations
- [ ] Context retention across conversation

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: TTS Announcements | 2-3 days | None |
| Phase 2: Voice Commands | 3-5 days | Phase 1 complete |
| Phase 3: Natural Conversations | 1-2 weeks | Phase 2 complete |
| Phase 4: Advanced Audio | 2-3 weeks | Phase 3 complete |

**Total:** 4-6 weeks for full implementation

**Quick Win:** Phase 1 only (TTS announcements) = 2-3 days

---

## Next Steps

### Immediate Actions
1. ✅ Review this plan with stakeholders
2. ⬜ Decide on initial scope (recommend Phase 1 only to start)
3. ⬜ Set up development environment with voice dependencies
4. ⬜ Choose TTS provider (recommend OpenAI for consistency)
5. ⬜ Create voice feature branch in git
6. ⬜ Implement basic VoiceManager class
7. ⬜ Test voice connection on development server

### Research Tasks
- [ ] Review Discord.py voice documentation
- [ ] Test OpenAI TTS voice quality
- [ ] Evaluate ElevenLabs trial for comparison
- [ ] Benchmark audio processing latency
- [ ] Research voice activity detection accuracy

---

## Resources & References

### Documentation
- [Discord.py Voice Support](https://discordpy.readthedocs.io/en/stable/api.html#voice-related)
- [OpenAI Text-to-Speech API](https://platform.openai.com/docs/guides/text-to-speech)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
- [FFmpeg Audio Processing](https://ffmpeg.org/documentation.html)

### Example Projects
- [Discord Music Bot Template](https://github.com/Rapptz/discord.py/tree/master/examples)
- [Python Voice Assistant](https://github.com/Leon-Sander/Local-AI-Asstant-with-GPT-4Vision-STT-TTS)

### Community Resources
- Discord.py Server: https://discord.gg/dpy
- Discord API Server: https://discord.gg/discord-api

---

## Conclusion

Voice integration would significantly enhance Megapig's capabilities, particularly for sports game announcements and interactive experiences. The phased approach allows for incremental development and testing, with clear milestones and manageable complexity.

**Recommended Starting Point:**
Phase 1 (TTS Announcements) - Achieves primary goal with minimal complexity, can be implemented in 2-3 days, and provides immediate value for sports notifications.

**Long-term Vision:**
Full voice conversation capability (Phase 3) would make Megapig one of the most advanced Discord bots available, offering natural voice interactions that feel like talking to a real assistant.

---

*Last Updated: 2025-09-30*
*Author: Claude (Sonnet 4.5)*
*Status: Planning Phase*