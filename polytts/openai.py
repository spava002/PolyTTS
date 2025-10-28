import os
from typing import Generator, Any

from .base import TTSProvider
from .types import AudioData

class OpenAITTS(TTSProvider):
    SAMPLE_RATE = 24000

    def __init__(self, api_key: str | None = None):
        """
        Initialize OpenAI TTS provider.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
        """
        super().__init__(api_key)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI is not installed. Install with: pip install polytts[openai]"
            )
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Provide it via the api_key parameter "
                "or set the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key)

    def get_sample_rate(self) -> int:
        """Get the sample rate for OpenAI TTS (24000 Hz)."""
        return self.SAMPLE_RATE

    def run(
        self, 
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        response_format: str = "pcm",
        **kwargs: Any
    ) -> AudioData:
        """
        Generate speech from text using OpenAI TTS.
        
        Args:
            text: The text to convert to speech (max 4096 characters)
            voice: Voice to use. Options: alloy, ash, ballad, coral, echo, fable,
                onyx, nova, sage, shimmer, verse, marin, cedar (default: "alloy")
            model: Model to use. Options: tts-1, tts-1-hd, gpt-4o-mini-tts (default: "tts-1")
            response_format: Output audio format. Options: mp3, opus, aac, flac, 
                wav, pcm (default: "pcm")
            **kwargs: Additional parameters
                
                For complete API reference:
                https://platform.openai.com/docs/api-reference/audio/createSpeech
            
        Returns:
            AudioData object containing the generated audio
            
        Example:
            >>> tts = OpenAITTS()
            >>> audio = tts.run("Hello world")
        """
        response = self.client.audio.speech.create(
            input=text,
            model=model,
            voice=voice,
            response_format=response_format,
            **kwargs
        )

        data = response.content
        sample_rate = self.get_sample_rate()
        return AudioData(data, sample_rate, response_format)

    def stream(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        response_format: str = "pcm",
        **kwargs: Any
    ) -> Generator[AudioData, None, None]:
        """
        Stream speech generation from text using OpenAI TTS.
        
        Args:
            text: The text to convert to speech (max 4096 characters)
            voice: Voice to use. Options: alloy, ash, ballad, coral, echo, fable,
                onyx, nova, sage, shimmer, verse, marin, cedar (default: "alloy")
            model: Model to use. Options: tts-1, tts-1-hd, gpt-4o-mini-tts (default: "tts-1")
            response_format: Output audio format. Options: mp3, opus, aac, flac, 
                wav, pcm (default: "pcm")
            **kwargs: Additional parameters
                
                For complete API reference:
                https://platform.openai.com/docs/api-reference/audio/createSpeech
            
        Yields:
            AudioData objects containing chunks of generated audio
            
        Example:
            >>> tts = OpenAITTS()
            >>> for chunk in tts.stream("Hello world"):
            ...     # Process each chunk in real-time
            ...     play_audio(chunk)
        """
        sample_rate = self.get_sample_rate()
        buffer = b""

        with self.client.audio.speech.with_streaming_response.create(
            input=text,
            model=model,
            voice=voice,
            response_format=response_format,
            **kwargs
        ) as response:
            for data in response.iter_bytes():
                # Ensure we have even number of bytes
                combined = buffer + data
                
                if len(combined) % 2 == 1:
                    buffer = combined[-1:]
                    combined = combined[:-1]
                else:
                    buffer = b""
                
                if len(combined) > 0:
                    yield AudioData(
                        data=combined, 
                        sample_rate=sample_rate, 
                        encoded_format=response_format
                    )
            
            if buffer:
                yield AudioData(
                    data=buffer + b"\x00", 
                    sample_rate=sample_rate, 
                    encoded_format=response_format
                )