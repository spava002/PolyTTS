import pytest
import numpy as np
from typing import Any

from polytts.audio import AudioData
from polytts.constants import AudioFormat

class TestAudioDataInit:
    """Test AudioData initialization."""

    def test_init_with_numpy_raw(self):
        """Test creating AudioData with numpy array."""
        audio = np.array([1, 2, 3], dtype=np.int16)
        audio_data = AudioData(audio, 22050, "raw")
        
        assert audio_data.sample_rate == 22050
        assert audio_data.encoded_format == "raw"
        assert audio_data.is_numpy
    
    def test_init_with_bytes_pcm(self):
        """Test creating AudioData with bytes."""
        audio_data = AudioData(b"12345678", 22050, "pcm")
        
        assert audio_data.sample_rate == 22050
        assert audio_data.encoded_format == "pcm"
        assert audio_data.is_bytes

class TestAudioDataPostInitValidation:
    """Test AudioData post-initialization validation."""

    def test_format_mismatch(self):
        """Test that numpy must use 'raw' and bytes must use encoded formats."""
        with pytest.raises(ValueError):
            AudioData(np.array([1, 2, 3]), 22050, "pcm")
        with pytest.raises(ValueError):
            AudioData(b"12345678", 22050, "raw")

    @pytest.mark.parametrize("invalid_data", [
        b"", np.array([]), None, "string", {}, 1.0, 123
    ])
    def test_invalid_data(self, invalid_data: Any):
        """Test that invalid data are rejected."""
        with pytest.raises((TypeError, ValueError)):
            AudioData(invalid_data, 22050, "raw")

    @pytest.mark.parametrize("invalid_sample_rate", [
        -1, 0, None, [], "22050", 123.5
    ])
    def test_invalid_sample_rate(self, invalid_sample_rate: Any):
        """Test that invalid sample rates are rejected."""
        with pytest.raises((TypeError, ValueError)):
            AudioData(np.array([1, 2, 3]), invalid_sample_rate, "raw")

    @pytest.mark.parametrize("invalid_encoded_format", [
        "ogg", "flac", "", None, 123, [], {}, 1.0
    ])
    def test_invalid_encoded_format(self, invalid_encoded_format: Any):
        """Test that invalid encoded formats are rejected."""
        with pytest.raises(ValueError):
            AudioData(np.array([1, 2, 3]), 22050, invalid_encoded_format)

class TestAudioDataProperties:
    """Test AudioData properties."""

    def test_is_numpy(self):
        """Test that AudioData is numpy."""
        audio = AudioData(np.array([1, 2, 3]), 22050, "raw")
        assert audio.is_numpy
        assert not audio.is_bytes

    def test_is_bytes(self):
        """Test that AudioData is bytes."""
        audio = AudioData(b"1234567890", 22050, "pcm")
        assert audio.is_bytes
        assert not audio.is_numpy

    def test_dtype(self):
        """Test that AudioData dtype is correct."""
        audio = AudioData(np.array([1, 2, 3], dtype=np.int16), 22050, "raw")
        assert audio.dtype == np.int16

    @pytest.mark.parametrize("encoded_format", [
        "raw",
        "pcm",
        "mp3",
        "wav"
    ])
    def test_duration(
        self,
        test_audio_data: tuple[int, dict],
        encoded_format: AudioFormat
    ):
        """Test that AudioData duration is correct."""
        sample_rate, audio_formats = test_audio_data
        original_audio = audio_formats["raw"]
        expected_duration = len(original_audio) / sample_rate

        audio = audio_formats[encoded_format]

        audio = AudioData(audio, sample_rate, encoded_format)
        if encoded_format == "mp3":
            assert audio.duration == pytest.approx(expected_duration, rel=0.1)
        else:
            assert audio.duration == pytest.approx(expected_duration, rel=0.01)