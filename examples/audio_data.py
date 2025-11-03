from polytts import ElevenLabsTTS

# Initialize the provider
tts = ElevenLabsTTS(api_key="your-api-key")

audio = tts.run("Hello, world!")

# View all of the audio data attributes
print(audio)

# Or access specific attributes
print(audio.data[:10])
print(audio.sample_rate)
print(audio.encoded_format)
print(audio.dtype)
print(audio.duration)

# You can also freely convert the audio data to different formats
pcm_bytes = audio.as_bytes("pcm")
wav_bytes = audio.as_bytes("wav")
mp3_bytes = audio.as_bytes("mp3")

numpy_audio_int16 = audio.as_numpy("int16")
numpy_audio_float32 = audio.as_numpy("float32")