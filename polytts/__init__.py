"""
PolyTTS - A unified interface for multiple TTS providers with seamless hotswapping.

Provides a consistent API across cloud-based and local TTS models.
"""

# Types
from .types import AudioData

# Cloud/API providers
from .openai import OpenAITTS
from .elevenlabs import ElevenLabsTTS
from .fishaudio import FishAudioTTS

# Local models
from .kokoro import KokoroTTS
from .gptsovits import GPTSovitsTTS

__version__ = "0.1.0"

__all__ = [
    "AudioData",
    "OpenAITTS",
    "ElevenLabsTTS",
    "FishAudioTTS",
    "KokoroTTS",
    "GPTSovitsTTS",
]