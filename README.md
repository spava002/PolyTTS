# PolyTTS

**A unified Python interface for multiple Text-to-Speech providers with seamless hotswapping.**

PolyTTS wraps various TTS providers behind a unified API:

- **One Interface, Many Providers**: Same code works regardless what provider you choose
- **Seamless Hotswapping**: Change providers by changing a single line
- **Smart Audio Handling**: Automatic conversion between bytes and numpy arrays
- **Cloud & Local Support**: Use cloud APIs or run models locally
- **Streaming Ready**: Real-time audio generation where supported

All providers return the same `AudioData` object with consistent conversion methods, so your downstream code stays the same regardless of which TTS you're using.

## Installation

```bash
# Basic installation
pip install polytts

# With specific providers
pip install polytts[openai]
pip install polytts[elevenlabs]
pip install polytts[fishaudio]
pip install polytts[kokoro]
pip install polytts[gptsovits]

# With all providers
pip install polytts[all]

# Note: If using UV, URL dependencies (like GPT-SoVITS) must be installed separately:
uv pip install git+https://github.com/spava002/GPT-SoVITS-Streaming.git

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Quick Example

```python
import soundfile as sf
from polytts import ElevenLabsTTS, KokoroTTS

# Start with a cloud provider
tts = ElevenLabsTTS(api_key="your-api-key")
audio = tts.run("Hello, world!")
sf.write("output.wav", audio.as_numpy(), audio.sample_rate)

# Switch to a local model - just change one line!
tts = KokoroTTS(lang_code="en-us")
audio = tts.run("Hello, world!")
sf.write("output.wav", audio.as_numpy(), audio.sample_rate)
```

## Supported Providers

| Provider | Type | Streaming | Voice Cloning |
|----------|------|-----------|---------------|
| OpenAI | Cloud | ✅ | ❌ |
| ElevenLabs | Cloud | ✅ | ✅ |
| Fish Audio | Cloud | ✅ | ✅ |
| Kokoro | Local | ✅ | ❌ |
| GPT-SoVITS | Local | ✅ | ✅ |

## Examples

Check out the [`examples/`](examples/) directory for complete working examples:

- **[`synthesis.py`](examples/synthesis.py)** - Basic text-to-speech generation
- **[`stream_synthesis.py`](examples/stream_synthesis.py)** - Real-time streaming audio
- **[`audio_data.py`](examples/audio_data.py)** - Working with AudioData conversions

## AudioData API

All providers return an `AudioData` object that makes format conversion trivial:

```python
audio = tts.run("Hello!")

# Access metadata
print(audio.sample_rate)
print(audio.duration)
print(audio.dtype)

# Convert formats
numpy_array = audio.as_numpy("float32")
pcm_bytes = audio.as_bytes("pcm")
wav_bytes = audio.as_bytes("wav")
```

## Contributing

Contributions are welcome! Whether it's:

- Bug fixes
- Documentation improvements
- **New TTS providers**

To add a new provider, check out the existing implementations in `polytts/`
Feel free to open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.