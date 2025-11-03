import numpy as np
from functools import cached_property
from dataclasses import dataclass, field
from typing import Literal, get_args

from .codecs import AudioConverter

EncodedBytesFormat = Literal["pcm", "mp3", "wav"]
EncodedNumpyFormat = Literal["raw"]

DType = Literal["int16", "int32", "float16", "float32"]

@dataclass
class AudioData:
    """
    Container for audio data with metadata and conversion utilities.
    
    Stores audio as either bytes (encoded formats like PCM, WAV, MP3) or
    numpy arrays (raw samples). Provides methods to convert between formats
    and data types.
    
    Attributes:
        data: Raw audio data as bytes or numpy array
        sample_rate: Audio sample rate in Hz
        encoded_format: Format of the audio data any of "pcm", "mp3", "wav", "raw"
        dtype: Numpy dtype if data is numpy array, None otherwise
        duration: Audio duration in seconds
    """
    data: bytes | np.ndarray
    sample_rate: int
    encoded_format: EncodedBytesFormat | EncodedNumpyFormat

    def __post_init__(self):
        """Validate inputs after dataclass initialization."""
        self._validate_inputs()

    def __repr__(self) -> str:
        """Custom repr that properly shows computed properties."""
        data_preview = f"<{len(self.data)} bytes>" if self.is_bytes else f"<array shape={self.data.shape}>"
        return (
            f"AudioData("
            f"data={data_preview}, "
            f"sample_rate={self.sample_rate}, "
            f"encoded_format='{self.encoded_format}', "
            f"dtype={self.dtype}, "
            f"duration={self.duration:.2f}s"
            f")"
        )

    @property
    def is_numpy(self) -> bool:
        """Check if audio data is stored as numpy array."""
        return isinstance(self.data, np.ndarray)

    @property
    def is_bytes(self) -> bool:
        """Check if audio data is stored as bytes."""
        return isinstance(self.data, bytes)

    @property
    def dtype(self) -> np.dtype | None:
        """Get numpy dtype of the data, or None if data is bytes."""
        if self.is_numpy:
            return self.data.dtype
        return None

    @cached_property
    def duration(self) -> float:
        """Get audio duration in seconds."""
        if self.encoded_format == "raw":
            return len(self.data) / self.sample_rate
        elif self.encoded_format == "pcm":
            num_samples = len(self.data) // 2
            return num_samples / self.sample_rate
        elif self.encoded_format == "mp3":
            try:
                import io
                from mutagen.mp3 import MP3
                audio = MP3(io.BytesIO(self.data))
                return audio.info.length
            except ImportError:
                raise ImportError(
                    "MP3 duration requires 'mutagen'. "
                    "Install with: pip install mutagen"
                )
        elif self.encoded_format == "wav":
            import io
            import wave
            with wave.open(io.BytesIO(self.data), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)

    def _validate_inputs(self):
        """Validate all input parameters."""
        self._validate_data()
        self._validate_sample_rate()
        self._validate_encoded_format()

    def _validate_data(self):
        """Validate that data is bytes or numpy array and not empty."""
        if not isinstance(self.data, (bytes, np.ndarray)):
            raise TypeError(
                f"Data must be bytes or numpy array, got {type(self.data).__name__}"
            )
        
        if len(self.data) == 0:
            raise ValueError("Data must not be empty")

    def _validate_sample_rate(self):
        """Validate that sample_rate is a positive integer."""
        if not isinstance(self.sample_rate, int):
            raise TypeError(
                f"Sample rate must be an integer, got {type(self.sample_rate).__name__}"
            )
        
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be greater than 0")

    def _validate_encoded_format(self):
        """Validate that encoded_format matches the data type bytes or numpy."""
        if self.is_numpy and self.encoded_format not in get_args(EncodedNumpyFormat):
            raise ValueError(
                f"Invalid encoded format for numpy array: {self.encoded_format}. "
                f"The only valid format is {get_args(EncodedNumpyFormat)}"
            )
        
        if self.is_bytes and self.encoded_format not in get_args(EncodedBytesFormat):
            raise ValueError(
                f"Invalid encoded format for bytes: {self.encoded_format}. "
                f"Valid formats are: {get_args(EncodedBytesFormat)}"
            )

    def as_bytes(self, output_format: EncodedBytesFormat = "pcm") -> bytes:
        """
        Convert audio data to bytes in specified format.
        
        Converts between PCM, WAV, and MP3 formats. Always returns mono audio.
        
        Args:
            output_format: Target format any of "pcm", "wav", or "mp3"
        
        Returns:
            Audio data as bytes in the specified format
        
        Examples:
            >>> audio = AudioData(numpy_array, 24000, "raw")
            >>> pcm_bytes = audio.as_bytes("pcm")
            >>> wav_bytes = audio.as_bytes("wav")
        """
        self._validate_output_format(output_format)
        return AudioConverter.to_bytes(self, output_format)

    def as_numpy(self, target_dtype: DType = "float32") -> np.ndarray:
        """
        Convert audio data to numpy array with specified dtype.
        
        Decodes encoded formats (PCM, WAV, MP3) to raw samples and converts
        to the target dtype with proper scaling between int and float types.
        
        Args:
            target_dtype: Target numpy dtype any of "float32", "float16", "int16", or "int32"
        
        Returns:
            Numpy array with mono audio samples in target dtype
        
        Examples:
            >>> audio = AudioData(pcm_bytes, 24000, "pcm")
            >>> samples_float32 = audio.as_numpy("float32")
            >>> samples_int16 = audio.as_numpy("int16")
        """
        self._validate_target_dtype(target_dtype)
        return AudioConverter.to_numpy(self, target_dtype)

    def _validate_output_format(self, output_format: EncodedBytesFormat):
        """Validate that output_format is one of the supported byte formats."""
        if output_format not in get_args(EncodedBytesFormat):
            raise ValueError(
                f"Invalid output format: {output_format}. "
                f"Valid formats are: {get_args(EncodedBytesFormat)}"
            )

    def _validate_target_dtype(self, target_dtype: DType):
        """Validate that target_dtype is one of the supported types."""
        if target_dtype not in get_args(DType):
            raise ValueError(
                f"Invalid target dtype: {target_dtype}. "
                f"Valid dtypes are: {get_args(DType)}"
            )