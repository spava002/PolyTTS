import os
from typing import Generator, Any

from .base import TTSProvider
from .types import AudioData

class FishAudioTTS(TTSProvider):
    SAMPLE_RATE = 44100

    def __init__(self, api_key: str | None = None):
        """
        Initialize Fish Audio TTS provider.
        
        Args:
            api_key: Fish Audio API key. If None, will try to get from FISHAUDIO_API_KEY env var.
        """
        super().__init__(api_key)
        
        try:
            from fish_audio_sdk import Session
        except ImportError:
            raise ImportError(
                "FishAudio is not installed. Install with: pip install polytts[fishaudio]"
            )
        
        api_key = self.api_key or os.getenv("FISHAUDIO_API_KEY")
        if not api_key:
            raise ValueError(
                "Fish Audio API key is required. Provide it via the api_key parameter "
                "or set the FISHAUDIO_API_KEY environment variable."
            )
        self.client = Session(apikey=api_key)

    def get_sample_rate(self) -> int:
        """Get the sample rate for Fish Audio TTS (44100 Hz)."""
        return self.SAMPLE_RATE

    def run(
        self, 
        text: str,
        reference_id: str | None = None,
        references: list | None = None,
        response_format: str = "pcm",
        **kwargs: Any
    ) -> AudioData:
        """
        Generate speech from text using Fish Audio TTS.
        
        Args:
            text: The text to convert to speech
            reference_id: Reference voice ID for voice cloning (from Fish Audio dashboard)
            references: List of ReferenceAudio objects for custom voice cloning
            response_format: Output audio format. Options: wav, pcm, mp3 (default: "mp3")
            **kwargs: Additional parameters
                
                Note: Do not pass a 'prosody' object. Use 'speed' and 'volume' parameters instead.

                For complete API reference:
                https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech
            
        Returns:
            AudioData object containing the generated audio
            
        Example:
            >>> tts = FishAudioTTS()
            >>> audio = tts.run("Hello world")
        """
        from fish_audio_sdk import TTSRequest, Prosody

        if references is None:
            references = []

        speed = kwargs.pop("speed", 1.0)
        volume = kwargs.pop("volume", 0.0)
        prosody = Prosody(speed=speed, volume=volume)

        sample_rate = self.get_sample_rate()
        response = self.client.tts(TTSRequest(
            text=text,
            format=response_format,
            reference_id=reference_id,
            references=references,
            prosody=prosody,
            **kwargs
        ))

        data = b"".join(response)
        return AudioData(data=data, sample_rate=sample_rate, encoded_format=response_format)

    def stream(
        self,
        text: str,
        reference_id: str | None = None,
        references: list | None = None,
        response_format: str = "pcm",
        **kwargs: Any
    ) -> Generator[AudioData, None, None]:
        """
        Stream speech generation from text using Fish Audio TTS.
        
        Args:
            text: The text to convert to speech
            reference_id: Reference voice ID for voice cloning (from Fish Audio dashboard)
            references: List of ReferenceAudio objects for custom voice cloning
            response_format: Output audio format. Options: wav, pcm, mp3 (default: "mp3")
            **kwargs: Additional parameters
                
                Note: Do not pass a 'prosody' object. Use 'speed' and 'volume' parameters instead.

                For complete API reference:
                https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech
            
        Yields:
            AudioData objects containing chunks of generated audio
            
        Example:
            >>> tts = FishAudioTTS()
            >>> for chunk in tts.stream("Hello world"):
            ...     # Process each chunk in real-time
            ...     play_audio(chunk)
        """
        from fish_audio_sdk import TTSRequest, Prosody

        if references is None:
            references = []

        speed = kwargs.pop("speed", 1.0)
        volume = kwargs.pop("volume", 0.0)
        prosody = Prosody(speed=speed, volume=volume)

        sample_rate = self.get_sample_rate()
        response = self.client.tts(TTSRequest(
            text=text,
            format=response_format,
            sample_rate=sample_rate,
            reference_id=reference_id,
            references=references,
            prosody=prosody,
            **kwargs
        ))

        for data in response:
            yield AudioData(data=data, sample_rate=sample_rate, encoded_format=response_format)