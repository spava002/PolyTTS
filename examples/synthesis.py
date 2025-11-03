import soundfile as sf

from polytts import ElevenLabsTTS

# Initialize the provider
tts = ElevenLabsTTS(api_key="your-api-key")

# Generate speech
audio = tts.run("Hello, world!")

# Save to file
sf.write("output.wav", audio.as_numpy(), audio.sample_rate)