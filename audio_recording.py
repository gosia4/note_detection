import librosa
import pyaudio
import numpy as np
import wave
def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


def record_audio_pyaudio(filename, duration=10, fs=44100):
    print("Rozpocznij nagrywanie...")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)

    frames = []
    for i in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Zakończono nagrywanie.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    recorded_data = np.concatenate(frames)

    # Zapisz nagranie do pliku
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(recorded_data.tobytes())

    print(f"Nagranie zapisane w pliku: {filename}")
    return recorded_data


# record_audio_pyaudio("user_sequences/jaillet73.wav", 5)