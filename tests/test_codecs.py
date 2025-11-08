from typing import Any
import pytest, filetype
import numpy as np

from polytts.codecs import AudioConverter
from polytts.constants import AudioFormat, EncodedBytesFormat, DType

class TestConversions:
    """Test audio conversions to bytes and numpy arrays."""

    @pytest.mark.parametrize("encoded_format, output_format", [
        # Same format conversions
        ("pcm", "pcm"),
        ("wav", "wav"),
        ("mp3", "mp3"),
        # Cross format conversions
        ("raw", "pcm"),
        ("raw", "wav"),
        ("raw", "mp3"),
        ("pcm", "wav"),
        ("pcm", "mp3"),
        ("wav", "pcm"),
        ("wav", "mp3"),
        ("mp3", "pcm"),
        ("mp3", "wav")
    ])
    def test_conversions_to_bytes(
        self,
        test_audio_data: tuple[int, dict],
        encoded_format: AudioFormat,
        output_format: EncodedBytesFormat
    ):
        """Test conversions to byte formats."""
        sample_rate, audio_formats = test_audio_data
        audio = audio_formats[encoded_format]

        audio = AudioConverter.to_bytes(audio, sample_rate, encoded_format, output_format)

        # Verify the bytes output format
        bytes_type = filetype.guess(audio)
        match output_format:
            case "wav":
                assert bytes_type is not None
                assert bytes_type.extension == "wav"
                assert bytes_type.mime == "audio/x-wav"
            case "mp3":
                assert bytes_type is not None
                assert bytes_type.extension == "mp3"
                assert bytes_type.mime == "audio/mpeg"
            case "pcm":
                assert bytes_type is None

    @pytest.mark.parametrize("encoded_format, target_dtype", [
        ("raw", "int16"),
        ("raw", "int32"),
        ("raw", "float16"),
        ("raw", "float32"),
        ("pcm", "int16"),
        ("pcm", "int32"),
        ("pcm", "float16"),
        ("pcm", "float32"),
        ("wav", "int16"),
        ("wav", "int32"),
        ("wav", "float16"),
        ("wav", "float32"),
        ("mp3", "int16"),
        ("mp3", "int32"),
        ("mp3", "float16"),
        ("mp3", "float32")
    ])
    def test_conversions_to_numpy(
        self,
        test_audio_data: tuple[int, dict],
        encoded_format: AudioFormat,
        target_dtype: DType
    ):
        """Test conversions to numpy arrays."""
        _, audio_formats = test_audio_data
        audio = audio_formats[encoded_format]

        audio = AudioConverter.to_numpy(audio, encoded_format, target_dtype)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == target_dtype

class TestValidations:
    """Test input validations for audio conversions."""

    @pytest.mark.parametrize("invalid_data", [
        b"", np.array([]), None, "string", {}, 1.0, 123
    ])
    def test_invalid_data_to_bytes(
        self,
        invalid_data: Any
    ):
        """Test that invalid data types are rejected."""
        with pytest.raises((ValueError, TypeError)):
            AudioConverter.to_bytes(invalid_data, 22050, "raw", "pcm")

    @pytest.mark.parametrize("invalid_sample_rate", [
        -1, 0, None, [], "22050", 123.5
    ])
    def test_invalid_sample_rate_to_bytes(
        self,
        test_audio_data: tuple[int, dict],
        invalid_sample_rate: Any
    ):
        """Test that invalid sample rates are rejected."""
        _, audio_formats = test_audio_data
        audio = audio_formats["raw"]
        with pytest.raises((ValueError, TypeError)):
            AudioConverter.to_bytes(audio, invalid_sample_rate, "raw", "pcm")

    @pytest.mark.parametrize("invalid_encoded_format", [
        "ogg", "flac", "", None, 123, [], {}, 1.0
    ])
    def test_invalid_encoded_format_to_bytes(
        self,
        test_audio_data: tuple[int, dict],
        invalid_encoded_format: Any
    ):
        """Test that invalid encoded formats are rejected."""
        _, audio_formats = test_audio_data
        audio = audio_formats["raw"]
        with pytest.raises((ValueError, TypeError)):
            AudioConverter.to_bytes(audio, 22050, invalid_encoded_format, "pcm")

    @pytest.mark.parametrize("invalid_output_format", [
        "ogg", "flac", "", None, 123, [], {}, 1.0
    ])
    def test_invalid_output_format_to_bytes(self,
        test_audio_data: tuple[int, dict],
        invalid_output_format: Any
    ):
        """Test that invalid output formats are rejected."""
        _, audio_formats = test_audio_data
        audio = audio_formats["raw"]
        with pytest.raises((ValueError, TypeError)):
            AudioConverter.to_bytes(audio, 22050, "raw", invalid_output_format)

    @pytest.mark.parametrize("invalid_data", [
        b"", np.array([]), None, "string", {}, 1.0, 123
    ])
    def test_invalid_data_to_numpy(
        self,
        invalid_data: Any
    ):
        """Test that invalid data are rejected."""
        with pytest.raises((ValueError, TypeError)):
            AudioConverter.to_numpy(invalid_data, "raw", "float32")

    @pytest.mark.parametrize("invalid_encoded_format", [
        "ogg", "flac", "", None, 123, [], {}, 1.0
    ])
    def test_invalid_encoded_format_to_numpy(
        self,
        test_audio_data: tuple[int, dict],
        invalid_encoded_format: Any
    ):
        """Test that invalid encoded formats are rejected."""
        _, audio_formats = test_audio_data
        audio = audio_formats["raw"]

        with pytest.raises((ValueError, TypeError)):
            AudioConverter.to_numpy(audio, invalid_encoded_format, "float32")

    @pytest.mark.parametrize("invalid_target_dtype", [
        "int8", "int64", "float64", "uint8", "", None, 123, []
    ])
    def test_invalid_target_dtype_to_numpy(
        self,
        test_audio_data: tuple[int, dict],
        invalid_target_dtype: Any
    ):
        """Test that invalid target dtypes are rejected."""
        _, audio_formats = test_audio_data
        audio = audio_formats["raw"]

        with pytest.raises((ValueError, TypeError)):
            AudioConverter.to_numpy(audio, "raw", invalid_target_dtype)