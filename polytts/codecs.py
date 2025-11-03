import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import AudioData, EncodedBytesFormat, DType

class AudioConverter:
    """Handles encoding and decoding between audio formats."""
    
    @staticmethod
    def to_bytes(audio: 'AudioData', output_format: 'EncodedBytesFormat') -> bytes:
        """
        Convert AudioData to bytes in specified format.
        
        Args:
            audio: AudioData object to convert
            output_format: Target format any of "pcm", "wav", "mp3"
        
        Returns:
            Audio data as bytes in the specified format
        """
        if audio.is_bytes and audio.encoded_format == output_format:
            return audio.data
        
        if audio.is_numpy:
            samples = AudioConverter._numpy_to_int16(audio.data)
        else:
            samples = AudioConverter._decode_to_numpy(audio)
        
        if output_format == "pcm":
            return AudioConverter._encode_pcm(samples, audio.sample_rate)
        elif output_format == "wav":
            return AudioConverter._encode_wav(samples, audio.sample_rate)
        elif output_format == "mp3":
            return AudioConverter._encode_mp3(samples, audio.sample_rate)
    
    @staticmethod
    def to_numpy(audio: 'AudioData', target_dtype: 'DType') -> np.ndarray:
        """
        Convert AudioData to numpy array with specified dtype.
        
        Args:
            audio: AudioData object to convert
            target_dtype: Target numpy dtype any of "float32", "float16", "int16", or "int32"
        
        Returns:
            Audio data as numpy array in target dtype
        """
        target_dtype = np.dtype(target_dtype)

        if audio.is_numpy:
            data = audio.data
        else:
            data = AudioConverter._decode_to_numpy(audio)
        
        if data.dtype == target_dtype:
            return data
        return AudioConverter._convert_dtype(data, target_dtype)
    
    @staticmethod
    def _numpy_to_int16(data: np.ndarray) -> np.ndarray:
        """Convert numpy array to int16 samples."""
        if data.dtype in (np.float16, np.float32):
            return (data * 32767.0).astype(np.int16)
        else:
            return data.astype(np.int16)
    
    @staticmethod
    def _convert_dtype(data: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
        """Convert numpy array to target dtype with proper scaling."""
        current_is_float = data.dtype in (np.float16, np.float32)
        target_is_float = target_dtype in (np.float16, np.float32)
        
        if current_is_float and not target_is_float:
            return (data * 32767.0).astype(target_dtype)
        elif not current_is_float and target_is_float:
            return data.astype(target_dtype) / 32768.0
        else:
            return data.astype(target_dtype)
    
    @staticmethod
    def _decode_to_numpy(audio: 'AudioData') -> np.ndarray:
        """Decode bytes to int16 mono numpy array."""
        if audio.encoded_format == "pcm":
            return AudioConverter._decode_pcm(audio)
        elif audio.encoded_format == "wav":
            return AudioConverter._decode_wav(audio)
        elif audio.encoded_format == "mp3":
            return AudioConverter._decode_mp3(audio)
        else:
            raise ValueError(f"Cannot decode format: {audio.encoded_format}")
    
    @staticmethod
    def _decode_pcm(audio: 'AudioData') -> np.ndarray:
        """Decode PCM bytes to int16 numpy array."""
        return np.frombuffer(audio.data, dtype=np.int16)
    
    @staticmethod
    def _decode_wav(audio: 'AudioData') -> np.ndarray:
        """Decode WAV bytes to int16 numpy array."""
        import io
        import wave
        
        with wave.open(io.BytesIO(audio.data), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            return np.frombuffer(frames, dtype=np.int16)
    
    @staticmethod
    def _decode_mp3(audio: 'AudioData') -> np.ndarray:
        """Decode MP3 bytes to int16 numpy array."""
        try:
            import io
            from pydub import AudioSegment
            
            audio_seg = AudioSegment.from_mp3(io.BytesIO(audio.data))
            return np.array(audio_seg.get_array_of_samples(), dtype=np.int16)
        except ImportError:
            raise ImportError(
                "MP3 decoding requires 'pydub' and 'ffmpeg'. "
                "Install: pip install pydub"
            )
    
    @staticmethod
    def _encode_pcm(samples: np.ndarray, sample_rate: int) -> bytes:
        """Encode int16 samples to PCM format bytes."""
        return samples.tobytes()

    @staticmethod
    def _encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        """Encode int16 samples to WAV format bytes."""
        import io
        import wave
        
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
        
        return buffer.getvalue()
    
    @staticmethod
    def _encode_mp3(samples: np.ndarray, sample_rate: int) -> bytes:
        """Encode int16 samples to MP3 format bytes."""
        try:
            import io
            from pydub import AudioSegment
            
            audio = AudioSegment(
                data=samples.tobytes(),
                sample_width=2,
                frame_rate=sample_rate,
                channels=1
            )
            
            buffer = io.BytesIO()
            audio.export(buffer, format="mp3")
            return buffer.getvalue()
            
        except ImportError:
            raise ImportError(
                "MP3 encoding requires 'pydub' and 'ffmpeg'. "
                "Install: pip install pydub"
            )