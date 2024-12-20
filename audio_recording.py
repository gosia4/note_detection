import librosa
import pyaudio
import numpy as np
import wave
def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


def record_audio_pyaudio(filename, duration=10, fs=44100):
    print("The recording has started...")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)

    frames = []
    for i in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("The recording has finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    recorded_data = np.concatenate(frames)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(recorded_data.tobytes())

    print(f"The recording has been saved in the file: {filename}")
    return recorded_data


# def record_audio_pyaudio(filename, duration=10, fs=44100, device_index=None):
#     print("The recording has started...")
#
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16,
#                     channels=1,
#                     rate=fs,
#                     input=True,
#                     frames_per_buffer=1024,
#                     input_device_index=device_index)
#
#     frames = []
#     for i in range(0, int(fs / 1024 * duration)):
#         data = stream.read(1024)
#         frames.append(np.frombuffer(data, dtype=np.int16))
#
#     print("The recording has finished.")
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     recorded_data = np.concatenate(frames)
#
#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(fs)
#         wf.writeframes(recorded_data.tobytes())
#
#     print(f"The recording has been saved in the file: {filename}")
#     return recorded_data
#
#
# def list_audio_devices():
#     p = pyaudio.PyAudio()
#     for i in range(p.get_device_count()):
#         info = p.get_device_info_by_index(i)
#         print(f"Device {i}: {info['name']}")
#     p.terminate()



# record_audio_pyaudio("user_sequences/elec3.wav", 4)