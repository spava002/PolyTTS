import pytest
import numpy as np

from polytts.codecs import AudioConverter

@pytest.fixture(scope="session")
def test_audio_data() -> tuple[int, dict]:
    """Generate test audio data for all formats."""
    sample_rate = 22050
    # Exactly 1 second of audio
    audio = np.random.randint(-32768, 32768, size=sample_rate, dtype=np.int16)

    audio_formats = {
        "raw": audio,
        "pcm": AudioConverter.to_bytes(audio, sample_rate, "raw", "pcm"),
        "wav": AudioConverter.to_bytes(audio, sample_rate, "raw", "wav"),
        "mp3": AudioConverter.to_bytes(audio, sample_rate, "raw", "mp3"),
    }

    return sample_rate, audio_formats