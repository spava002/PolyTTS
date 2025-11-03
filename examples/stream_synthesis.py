import numpy as np
import soundfile as sf

from polytts import ElevenLabsTTS

# Initialize the provider
tts = ElevenLabsTTS(api_key="your-api-key")

# Collect audio chunks
audio_chunks = []
for chunk in tts.stream("Hello, world!"):
    audio_chunks.append(chunk.as_numpy("int16"))
    print(chunk)

# Save the audio
audio = np.concatenate(audio_chunks)
sf.write("output.wav", audio, tts.get_sample_rate())