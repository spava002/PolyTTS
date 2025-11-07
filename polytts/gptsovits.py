from typing import Generator, Any

from .base import TTSProvider
from .types import AudioData


class GPTSovitsTTS(TTSProvider):
    def __init__(self, model_version: str = "v2ProPlus", warmup_model: bool = True):
        """
        Initialize GPT-SoVITS TTS provider.

        Args:
            model_version: Model version to use. Options:
                "v2ProPlus", "v2Pro", "v2", "v1", "v3", "v4"
            warmup_model: Whether to warmup the model. Faster inference on consecutive calls
        """
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        except ImportError:
            raise ImportError(
                "GPTSoVITS is not installed. Install with one of:\n"
                "  - pip install polytts[gptsovits]  (recommended)\n"
                "  - Manual setup from: https://github.com/RVC-Boss/GPT-SoVITS"
            )

        self.client = TTS(
            TTS_Config.default_configs[model_version]
        )  # This wont work with CUDA
        if warmup_model:
            self.client.run("Warming up GPT-SoVITS model...")

    def get_sample_rate(self) -> int:
        return self.client.configs.sampling_rate

    def run(
        self,
        text: str,
        text_lang: str,
        ref_audio_path: str | None = None,
        **kwargs: Any,
    ) -> AudioData:
        """
        Generate speech from text using GPT-SoVITS.

        Args:
            text: The text to convert to speech
            
            text_lang: Language of the text. Options: "en", "zh", "ja", "ko", etc.
            
            ref_audio_path: Path to reference audio file for voice cloning.
                Should be 3-10 seconds of clear speech. If None, uses default reference.
            
            **kwargs: Additional parameters
                prompt_text: Transcript of reference audio for better cloning. Default: "".
                
                prompt_lang: Language of prompt text. Options: "en", "zh", "ja", "ko", etc. Default: "".
                
                aux_ref_audio_paths: Additional reference audios for tone fusion. Default: [].

                speed_factor: Speech speed multiplier. Default: 1.0
                
                For complete API reference:
                    If installed with polytts[gptsovits]: https://github.com/spava002/GPT-SoVITS-Streaming/blob/main/GPT_SoVITS/TTS_infer_pack/TTS.py#L1015
                    If installed from official: https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/TTS_infer_pack/TTS.py#L984

        Returns:
            AudioData object with generated audio

        Example:
            >>> tts = GPTSovitsTTS()
            >>> audio = tts.run("Hello world")
        """
        if ref_audio_path is None:
            ref_audio_path = "polytts/audio/audio1.wav"

        inputs = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            **kwargs,
        }

        response = self.client.run(inputs)

        sample_rate, data = next(response)
        return AudioData(
            data=data,
            sample_rate=sample_rate,
            encoded_format="raw"
        )

    def stream(
        self,
        text: str,
        text_lang: str,
        ref_audio_path: str | None = None,
        **kwargs: Any,
    ) -> Generator[AudioData, None, None]:
        """
        Stream speech generation from text using GPT-SoVITS.

        Args:
            text: The text to convert to speech
            
            text_lang: Language of the text. Options: "en", "zh", "ja", "ko", etc.
            
            ref_audio_path: Path to reference audio file for voice cloning.
                Should be 3-10 seconds of clear speech. If None, uses default reference.
            
            **kwargs: Additional parameters
                prompt_text: Transcript of reference audio for better cloning. Default: "".
                
                prompt_lang: Language of prompt text. Options: "en", "zh", "ja", "ko", etc. Default: "".
                
                aux_ref_audio_paths: Additional reference audios for tone fusion. Default: [].

                speed_factor: Speech speed multiplier. Default: 1.0
                
                For complete API reference:
                    If installed with polytts[gptsovits]: 
                    https://github.com/spava002/GPT-SoVITS-Streaming/blob/main/GPT_SoVITS/TTS_infer_pack/TTS.py#L1015
                    
                    If installed from official: 
                    https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/TTS_infer_pack/TTS.py#L984

        Yields:
            AudioData objects containing chunks of generated audio

        Example:
            >>> tts = GPTSovitsTTS()
            >>> for chunk in tts.stream("Hello world"):
            ...     print(chunk)
        """
        if ref_audio_path is None:
            ref_audio_path = "polytts/audio/audio1.wav"

        inputs = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "return_fragment": True,
            "parallel_infer": False,
            **kwargs,
        }

        for sample_rate, data in self.client.run(inputs):
            yield AudioData(
                data=data,
                sample_rate=sample_rate,
                encoded_format="raw",
            )
