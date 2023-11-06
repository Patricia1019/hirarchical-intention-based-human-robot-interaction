import deepspeech
import wave
import numpy as np
from pathlib import Path
import pyaudio
FILE_DIR = Path(__file__).parent

# Initialize the DeepSpeech model with the downloaded model and scorer
model = deepspeech.Model(f'{FILE_DIR}/deepspeech-0.9.3-models.pbmm')
model.enableExternalScorer(f'{FILE_DIR}/deepspeech-0.9.3-models.scorer')

import noisereduce as nr
import soundfile as sf

# Define parameters
sample_rate = 16000  # Adjust as needed
chunk_size = 1024

# Create a PyAudio stream for microphone input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                output=True,  # Enable audio output
                frames_per_buffer=chunk_size)

# Initialize a noise profile with the first few seconds of input
noise_profile = np.zeros(chunk_size, dtype=np.float32)
for _ in range(10):  # Capture the first 10 chunks of audio as noise profile
    audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
    noise_profile += audio_data

noise_profile = noise_profile / 10  # Average the noise profile
ds_stream = model.createStream()
print("Noise profile captured. Start speaking.")

try:
    count = 0
    while True:
        audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
        # Apply noise reduction using the noise profile
        denoised_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)

        # # Play the denoised audio
        # stream.write(denoised_audio.tobytes())

        # # Process the denoised audio data as needed
        # # For this example, we transcribe the audio to text using DeepSpeech
        # text = model.stt()
        # # text = model.stt(audio_data)
        ds_stream.feedAudioContent(audio_data)
        partial_transcription = ds_stream.intermediateDecode()
        if len(partial_transcription) > count:
            count = len(partial_transcription)
            print(partial_transcription.split()[-1])

except KeyboardInterrupt:
    print("Stopped listening.")

# Close the PyAudio stream
stream.stop_stream()
stream.close()
p.terminate()

print("Real-time noise reduction and transcription completed.")
