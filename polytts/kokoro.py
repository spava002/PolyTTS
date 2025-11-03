import numpy as np
from typing import Generator, Any

from .base import TTSProvider
from .types import AudioData


class KokoroTTS(TTSProvider):
    SAMPLE_RATE = 24000

    def __init__(self, lang_code: str = "a", device: str | None = None):
        """
        Initialize Kokoro TTS provider.

        Args:
            lang_code: Language code for G2P processing. Options:
                - "a" or "en-us": American English
                - "b" or "en-gb": British English
                - "e" or "es": Spanish (espeak-ng)
                - "f" or "fr-fr": French (espeak-ng)
                - "h" or "hi": Hindi (espeak-ng)
                - "i" or "it": Italian (espeak-ng)
                - "j" or "ja": Japanese (requires: pip install misaki[ja])
                - "p" or "pt-br": Brazilian Portuguese (espeak-ng)
                - "z" or "zh": Mandarin Chinese
            device: Device to run model on. Options: "cuda", "cpu", or
                None for auto-detect
        """
        super().__init__(api_key=None)

        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "Kokoro is not installed. Install with: pip install polytts[kokoro]"
            )

        self.client = KPipeline(lang_code=lang_code, device=device)

    def get_sample_rate(self) -> int:
        """Get the sample rate for Kokoro TTS (24000 Hz)."""
        return self.SAMPLE_RATE

    def run(
        self,
        text: str,
        voice: str = "af_heart",
        **kwargs: Any
    ) -> AudioData:
        """
        Generate speech from text using Kokoro TTS.

        Args:
            text: The text to convert to speech.
                Long texts are automatically chunked.
            voice: Voice identifier to use. Common voices:
                - "af_heart": Female voice (warm, clear)
                - "af_bella": Female voice (bright, energetic)
                - "af_jessica": Female voice (professional)
                - "af_sarah": Female voice (soft, gentle)
                - "am_adam": Male voice (deep, authoritative)
                - "am_michael": Male voice (clear, neutral)
                - Multiple voices can be blended: "af_heart, af_bella"
            **kwargs: Additional parameters

                For complete API reference:
                https://github.com/hexgrad/kokoro

        Returns:
            AudioData object with generated audio

        Example:
            >>> tts = KokoroTTS()
            >>> audio = tts.run("Hello world")
        """
        # Kokoro returns a generator, so we collect all chunks
        response = self.client(
            text=text,
            voice=voice,
            **kwargs
        )

        audio_chunks = []
        for _, _, audio in response:
            audio_chunks.append(audio.numpy())

        # Concatenate all chunks
        full_audio = (
            np.concatenate(audio_chunks)
            if audio_chunks
            else np.array([], dtype=np.float32)
        )

        return AudioData(
            data=full_audio,
            sample_rate=self.get_sample_rate(),
            encoded_format="raw"
        )

    def stream(
        self,
        text: str,
        voice: str = "af_heart",
        **kwargs: Any
    ) -> Generator[AudioData, None, None]:
        """
        Stream speech generation from text using Kokoro TTS.

        Args:
            text: The text to convert to speech.
                Long texts are automatically chunked.
            voice: Voice identifier to use. Common voices:
                - "af_heart": Female voice (warm, clear)
                - "af_bella": Female voice (bright, energetic)
                - "af_jessica": Female voice (professional)
                - "af_sarah": Female voice (soft, gentle)
                - "am_adam": Male voice (deep, authoritative)
                - "am_michael": Male voice (clear, neutral)
                - Multiple voices can be blended: "af_heart, af_bella"
            **kwargs: Additional parameters

                For complete API reference:
                https://github.com/hexgrad/kokoro

        Yields:
            AudioData objects with chunks of audio

        Example:
            >>> tts = KokoroTTS()
            >>> for chunk in tts.stream("Hello world"):
            ...     # Process each chunk in real-time
            ...     play_audio(chunk)
        """
        response = self.client(
            text=text,
            voice=voice,
            **kwargs
        )

        sample_rate = self.get_sample_rate()
        for _, _, audio in response:
            yield AudioData(
                data=audio.numpy(),
                sample_rate=sample_rate,
                encoded_format="raw"
            )
