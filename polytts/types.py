import numpy as np
from dataclasses import dataclass
from typing import Literal

EncodedFormat = Literal["pcm", "mp3", "opus", "wav", "raw"]
DType = Literal["float32", "int16", "int32", "int64"]

@dataclass
class AudioData:
    data: bytes | np.ndarray
    sample_rate: int
    encoded_format: str  # For bytes: "pcm", "mp3", "opus", "wav", etc. For numpy: "raw"
    dtype: str | None = None  # For numpy arrays: "float32", "int16", etc. None for bytes

    # TODO: Add audio duration
    
    def __post_init__(self):
        """Validate that metadata matches actual data."""
        if self.is_numpy:
            # Validate dtype for numpy arrays
            if self.dtype is None:
                # Auto-detect dtype if not provided
                self.dtype = str(self.data.dtype)
            elif str(self.data.dtype) != self.dtype:
                raise ValueError(
                    f"dtype mismatch: AudioData.dtype is '{self.dtype}' but "
                    f"actual numpy array dtype is '{self.data.dtype}'"
                )
        else:
            # For bytes, dtype should be None
            if self.dtype is not None:
                raise ValueError(
                    f"dtype should be None for byte data, got '{self.dtype}'"
                )
    
    @property
    def is_numpy(self) -> bool:
        return isinstance(self.data, np.ndarray)
    
    @property
    def is_bytes(self) -> bool:
        return isinstance(self.data, bytes)
    
    def as_bytes(self) -> bytes:
        """Convert audio data to bytes (int16 PCM format)."""
        if self.is_bytes: 
            return self.data
        
        # Convert numpy to bytes (int16 format)
        if self.data.dtype == np.float32:
            # Float audio data (typically -1.0 to 1.0) -> int16
            data = (self.data * 32767.0).astype(np.int16)
        elif self.data.dtype == np.int16:
            data = self.data
        else:
            # Other int types -> convert to int16
            data = self.data.astype(np.int16)
        return data.tobytes()
    
    def as_numpy(self, target_dtype: str = "float32") -> np.ndarray:
        """
        Convert audio data to numpy array.
        
        Args:
            target_dtype: Target numpy dtype. Default is "float32" (normalized to -1.0 to 1.0)
        
        Returns:
            Numpy array with the specified dtype
        """
        if self.is_numpy:
            # Already numpy - convert to target dtype if needed
            if str(self.data.dtype) == target_dtype:
                return self.data
            
            if target_dtype == "float32":
                if self.data.dtype in (np.int16, np.int32):
                    return self.data.astype(np.float32) / 32768.0
                else:
                    return self.data.astype(np.float32)
            elif target_dtype == "int16":
                if self.data.dtype == np.float32:
                    return (self.data * 32767.0).astype(np.int16)
                else:
                    return self.data.astype(np.int16)
            else:
                return self.data.astype(np.dtype(target_dtype))
        
        # Convert bytes to numpy
        if self.encoded_format not in ("pcm", "raw"):
            raise ValueError(
                f"Cannot convert {self.encoded_format} format to numpy. "
                f"Only 'pcm' or 'raw' formats are supported for conversion."
            )

        # Assume int16 PCM data in bytes
        int16_array = np.frombuffer(self.data, dtype=np.int16)
        
        if target_dtype == "float32":
            return int16_array.astype(np.float32) / 32768.0
        elif target_dtype == "int16":
            return int16_array
        else:
            return int16_array.astype(np.dtype(target_dtype))