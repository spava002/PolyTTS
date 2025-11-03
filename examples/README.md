# PolyTTS Examples

This directory contains example scripts demonstrating various features of PolyTTS.

## Prerequisites

Install PolyTTS with at least one provider:

```bash
# For cloud providers
pip install polytts[openai]
pip install polytts[elevenlabs]

# For local providers
pip install polytts[kokoro]

# Or install all providers
pip install polytts[all]
```

Also install `soundfile` for saving audio:

```bash
pip install soundfile
```

## Examples

### 1. Basic Synthesis (`synthesis.py`)

The simplest example - generate speech and save to a file.

```bash
python synthesis.py
```

**What it shows:**
- Initialize a TTS provider
- Generate audio from text with `run()`
- Save audio using `soundfile`

### 2. Streaming Synthesis (`stream_synthesis.py`)

Generate audio in real-time chunks for faster response times.

```bash
python stream_synthesis.py
```

**What it shows:**
- Use `stream()` for chunk-by-chunk audio generation
- Process audio chunks as they arrive
- Concatenate chunks and save to file

### 3. AudioData Usage (`audio_data.py`)

Explore the `AudioData` object and its conversion capabilities.

```bash
python audio_data.py
```

**What it shows:**
- Access audio metadata (sample rate, duration, dtype)
- Convert between formats (PCM, WAV, MP3)
- Convert between dtypes (int16, float32, etc.)
- Work with both bytes and numpy arrays

## Setting Up API Keys

For cloud providers, you'll need API keys. Create a `.env` file:

```bash
# .env
OPENAI_API_KEY=your-openai-key
ELEVENLABS_API_KEY=your-elevenlabs-key
FISHAUDIO_API_KEY=your-fishaudio-key
```

Then load them in your script:

```python
from dotenv import load_dotenv
load_dotenv()

from polytts import ElevenLabsTTS

tts = ElevenLabsTTS()  # Reads from environment
```

Or load them in directly:

```python
from polytts import ElevenLabsTTS

tts = ElevenLabsTTS(api_key="your-api-key")
```

## Local Providers

Local providers like Kokoro and GPT-SoVITS don't require API keys:

```python
from polytts import KokoroTTS

# No API key needed!
tts = KokoroTTS(lang_code="en-us")
audio = tts.run("Hello, world!")
```