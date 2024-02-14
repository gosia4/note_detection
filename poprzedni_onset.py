

def detect_onsets_user_frequency(audio_data,  spectrogram_output=None, onsets_output=None, threshold_factor=0.02, plot=True, fs=44100):
    # Oblicz spektrogram
    f, t, Sxx = spectrogram(audio_data, fs=fs, nperseg=1024)
    # f - częstotliwości (osie y spektrogramu), t - czas (osie x spektrogramu), Sxx - spektrogram

    # Wyświetlenie spektrogramu, pcolormesh - do tworzenia plots jako mapa kolorów
    if plot:
        # plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='auto')  # usunięcie ujemnych wartości

        plt.title('Spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    if spectrogram_output:
        # plt.savefig(spectrogram_output)
        plt.imsave(spectrogram_output, 10 * np.log10(Sxx + 1e-10))

    onsets = []  # Lista przechowująca czasy detekcji onsetów
    current_spectrum = 0
    i = 0
    # Iteracja po kolumnach spektrogramu
    for i in range(1, len(t)):
        current_spectrum_values = []

        # Iteracja po wierszach spektrogramu, dla każdej kolumny oblicza potem wartość mean
        for row in Sxx:
            current_spectrum_values.append(row[i])

        # Srednią wartość spektralną
        average_spectrum_value = np.mean(current_spectrum_values)

        # próg dynamicznie
        threshold = threshold_factor * average_spectrum_value
        # threshold = threshold_factor * find_max_value(Sxx) # próg w oparciu o maksymalną wartość w spektrogramie

        # # Oblicz średnią wartość mocy spektralnej
        # average_spectrum_power = np.mean(np.abs(current_spectrum_values) ** 2)
        #
        # # Ustaw próg dynamicznie w oparciu o aktualną głośność
        # threshold = threshold_factor * average_spectrum_power


        # Sprawdź, czy aktualna wartość spektralna w danej kolumnie przekracza próg
        # i jednocześnie nie przekraczał go w poprzedniej
        if np.max(Sxx[:, i]) > threshold >= np.max(Sxx[:, i - 1]):
            onsets.append(t[i])

    # Wyświetlenie czasu onsetów
    if onsets:
        print("Onsets detected at the following times:")
        for onset_time in onsets:
            # i += 1
            print(f"{onset_time:.2f} seconds")

    # Wykres siły onsetów
    plt.figure()
    plt.plot(t, np.max(Sxx, axis=0), label='Spectral Power')
    plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Spectral Power and Onsets')
    plt.xlabel('Time [sec]')
    plt.ylabel('Spectral Power')
    plt.legend()
    plt.show()

    # if spectrogram_output:
    #     # plt.imsave(spectrogram_output, 10 * np.log10(Sxx + 1e-10))
    #     plt.imsave(spectrogram_output, plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='auto'))
    #     # plt.savefig(spectrogram_output)
    if onsets_output:
        with open(onsets_output, 'w') as file:
            for onset_time in onsets:
                file.write(f"{onset_time:.2f}\n")

    return onsets


# do znalezienia maksymalnej wartości w macierzy
def find_max_value(matrix):
    max_value = float('-inf')
    for row in matrix:
        for value in row:
            if value > max_value:
                max_value = value
    return max_value

# do znajdowania maksymalnej wartości w kolumnie macierzy
def custom_max(column):
    max_value = float('-inf')
    for value in column:
        if value > max_value:
            max_value = value
    return max_value


def record_audio_pyaudio(duration=5, fs=44100):
    print("Rozpocznij nagrywanie...")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)

    frames = [] # lista do przechowywania nagranych raemk dźwiękowych
    # jako rnage: czas trwania nagrywania, częstotliwość próbkowania i liczba próbek na ramkę
    for i in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.int16))  # Konwertuje dane z formatu bajtowego do tablicy numpy
                                                            # zawierającej liczby całkowite 16-bitowe

    print("Zakończono nagrywanie.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames) # łączenie ramek

def calculate_dtw_distance(onsets1, onsets2):
    # Zamień czasy onsetów na indeksy dla obliczeń DTW
    onsets1_idx = np.arange(len(onsets1)).reshape(-1, 1)
    onsets2_idx = np.arange(len(onsets2)).reshape(-1, 1)

    # Oblicz odległość DTW
    distance, path = fastdtw(onsets1_idx, onsets2_idx)

    return distance

# Przykładowe nagrania dźwięku od użytkownika za pomocą pyaudio
# user_audio1 = record_audio_pyaudio()
# user_audio2 = record_audio_pyaudio()
# user_audio3 = record_audio_pyaudio()
#
# # Detekcja onsetów z danych od użytkownika na podstawie częstotliwości
# onsets_user_frequency1 = detect_onsets_user_frequency(user_audio1, "s1.png", "o1.txt")
# onsets_user_frequency2 = detect_onsets_user_frequency(user_audio2, "s2.png", "o2.txt")
# onsets_user_frequency3 = detect_onsets_user_frequency(user_audio3, "s3.png", "o3.txt")
#
# # Oblicz odległość DTW między dwoma zestawami onsetów
# dtw_distance = calculate_dtw_distance(onsets_user_frequency1, onsets_user_frequency2)
# dtw_distance2 = calculate_dtw_distance(onsets_user_frequency1, onsets_user_frequency3)
# dtw_distance3 = calculate_dtw_distance(onsets_user_frequency3, onsets_user_frequency2)
#
# # Wyświetl wynik
# print(f"DTW Distance: {dtw_distance}")
# print("\n")
# print(f"DTW Distance: {dtw_distance2}")
# print("\n")
# print(f"DTW Distance: {dtw_distance3}")


# user_audio = record_audio_pyaudio()

# Detekcja onsetów z danych od użytkownika na podstawie częstotliwości
# onsets_user_frequency = detect_onsets_user_frequency(user_audio)
# onsets_user_frequency = detect_onsets_user_frequency(wave.open("C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AndreHolzapfel/bongo1.wav"))

# print("Onsets from user (frequency):", onsets_user_frequency)
# note_detect(wave.open("C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AndreHolzapfel/bongo1.wav"))

# do wywołania dwa utwory z bazy danych i zastoosowanie dtw
# wav_file = wave.open('C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/Pertusa/tiersen11.wav', 'rb')  # treshold 80
# wav_file2 = wave.open('C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AndreHolzapfel/bongo1.wav', 'rb')  # treshold 30
# frame_rate = wav_file.getframerate()
# num_frames = wav_file.getnframes()
# audio_data = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16)
# wav_file.close()
#
# onsets = detect_onsets_user_frequency(audio_data, "s1.png", "o1.txt", threshold_factor=80, plot=True, fs=frame_rate)
#
# frame_rate = wav_file2.getframerate()
# num_frames = wav_file2.getnframes()
# audio_data = np.frombuffer(wav_file2.readframes(num_frames), dtype=np.int16)
# wav_file2.close()
#
# onsets2 = detect_onsets_user_frequency(audio_data, "s2.png", "o2.txt",threshold_factor=30, plot=True, fs=frame_rate)
#
#
# dtw_distance2 = calculate_dtw_distance(onsets, onsets2)
# dtw_distance = dtw.dtw_distance(onsets, onsets2)
#
# print(f"DTW Distance: {dtw_distance}")
# print(f"DTW Distance: {dtw_distance2}")









# if __name__ == "__main__":
#     # file_name = "C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AndreHolzapfel/bongo1.wav"
#     file_name = "plik.wav"
#     audio_file = wave.open(file_name)
#     Detected_Result = note_detect(audio_file)
#     print("\n\tDetected Result = " + str(Detected_Result))
# def note_detect(audio_file_path):
#     audio_file = wave.open(audio_file_path, 'rb')  # mode binrary
#     threshold_factor = 0.778
#     file_length = audio_file.getnframes()
#     f_s = audio_file.getframerate()
#
#     # Read the entire audio file into an array
#     sound_frames = audio_file.readframes(file_length)
#     sound = np.frombuffer(sound_frames, dtype=np.int16)
#
#     plt.plot(sound)
#     plt.title('Audio Signal')
#     plt.show()
#
#     sound = np.divide(sound, float(2 ** 15))
#
#     plt.plot(sound)
#     plt.title('Normalized Signal')
#     plt.show()
#
#     # Onset detection
#     threshold = threshold_factor * np.max(sound)
#     onsets = []
#     onset_detected = False
#
#     for i in range(1, file_length):
#         if sound[i] > threshold >= sound[i - 1]:
#             onset_detected = True
#             onsets.append(i)
#
#     # Display time of onsets
#     if onsets:
#         print("Onsets detected at the following times:")
#         for onset_index in onsets:
#             time = onset_index / f_s
#             print(f"{time:.2f} seconds")
#     audio_file.close()
#
#     return onsets  # w której próbce dźwięku wystąpił onset
#
#
# # Nagrywanie dźwięku z mikrofonu za pomocą pyaudio
# def record_audio_pyaudio(duration=5, fs=44100):
#     print("Rozpocznij nagrywanie...")
#
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
#
#     frames = []
#     for i in range(0, int(fs / 1024 * duration)):
#         data = stream.read(1024)
#         frames.append(np.frombuffer(data, dtype=np.int16))
#
#     print("Zakończono nagrywanie.")
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
#     return np.concatenate(frames)
#
#
# # Przykładowe nagrywanie dźwięku od użytkownika
# user_audio = record_audio_pyaudio()
#
# # Detekcja onsetów w danych od użytkownika
# onsets_user = note_detect(user_audio)
#
# # Wyświetl czasy onsetów
# print("Onsets from user:", onsets_user)


# def detect_onsets_audio(audio_file_path):
#     # Wczytaj plik audio
#
#     onsets = note_detect(audio_file)
#     audio_file.close()
#
#     return onsets

#
# audio_file_path = "C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AndreHolzapfel/bongo1.wav"
#
# onsets_audio = note_detect(audio_file_path)
#
# print("Onsets from audio file:", onsets_audio)


# if __name__ == "__main__":
#     # file_name = "C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AndreHolzapfel/bongo1.wav"
#     file_name = "plik.wav"
#     audio_file = wave.open(file_name)
#     Detected_Result = note_detect(audio_file)
#     print("\n\tDetected Result = " + str(Detected_Result))

# import numpy as np
# import math
# import wave
# import struct
# import os
# import scipy.signal
#
#
# # Teams can add helper functions
# # Add all helper functions here
#
# ############################### Your Code Here #############################################
#
# def onset_detect(audio_file):
#     #   Instructions
#     #   ------------
#     #   Input 	:	audio_file -- a single test audio_file as input argument
#     #   Output	:	1. Onsets -- List of Float numbers corresponding
#     #							 to the Note Onsets (up to Two decimal places)
#     #				2. Detected_Notes -- List of string corresponding
#     #									 to the Detected Notes
#     #	Example	:	For Audio_1.wav file,
#     # 				Onsets = [0.00, 2.15, 4.30, 7.55]
#     #				Detected_Notes = ["F4", "B3", "C6", "A4"]
#
#     Onsets = []
#     Detected_Notes = []
#
#     # Add your code here
#     sampling_freq = 44100
#     window = 399
#     start = []
#     end = []
#
#     file_length = audio_file.getnframes()
#     sound = np.zeros(file_length)
#     for i in range(file_length):
#         data = audio_file.readframes(1)
#         data = struct.unpack("<h", data)
#         sound[i] = int(data[0])
#     sound = np.divide(sound, float(2 ** 15))
#     sound_square = np.square(sound)
#
#     array = [17.32, 19.45, 23.12, 25.96, 29.14,
#              34.65, 38.89, 46.25, 51.91, 58.27,
#              69.30, 77.78, 92.50, 103.83, 116.54,
#              138.59, 155.56, 185.00, 207.65, 233.08,
#              277.18, 311.13, 369.99, 415.30, 466.16,
#              554.37, 622.25, 739.99, 830.61, 932.33,
#              1108.73, 1244.51, 1479.98, 1661.22, 1864.66,
#              2217.46, 2489.02, 2959.96, 3322.44, 3729.31,
#              4434.92, 4978.03, 5919.91, 6644.88, 7458.62,
#              16.35, 18.35, 20.60, 21.83, 24.50, 27.50, 30.87,
#              32.70, 36.71, 41.20, 43.65, 49.00, 55.00, 61.74,
#              65.41, 73.42, 82.41, 87.31, 98.00, 110.00, 123.47,
#              130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,
#              261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,
#              523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,
#              1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53,
#              2093.00, 2349.32, 2637.02, 2793.83, 3135.96, 3520.00, 3951.07,
#              4186.01, 4698.63, 5274.04, 5587.65, 6271.93, 7040.00, 7902.13,
#              ]
#
#     notes = ['C#0', 'D#0', 'F#0', 'G#0', 'A#0',
#              'C#1', 'D#1', 'F#1', 'G#1', 'A#1',
#              'C#2', 'D#2', 'F#2', 'G#2', 'A#2',
#              'C#3', 'D#3', 'F#3', 'G#3', 'A#3',
#              'C#4', 'D#4', 'F#4', 'G#4', 'A#4',
#              'C#5', 'D#5', 'F#5', 'G#5', 'A#5',
#              'C#6', 'D#6', 'F#6', 'G#6', 'A#6',
#              'C#7', 'D#7', 'F#7', 'G#7', 'A#7',
#              'C#8', 'D#8', 'F#8', 'G#8', 'A#8',
#              'C0', 'D0', 'E0', 'F0', 'G0', 'A0', 'B0',
#              'C1', 'D1', 'E1', 'F1', 'G1', 'A1', 'B1',
#              'C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2',
#              'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
#              'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
#              'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5',
#              'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6',
#              'C7', 'D7', 'E7', 'F7', 'G7', 'A7', 'B7',
#              'C8', 'D8', 'E8', 'F8', 'G8', 'A8', 'B8',
#              ]
#
#     i = 0
#     xsum = []
#     c = 0
#     count = 0
#     while (i < (file_length) - window):
#         s = 0.00
#         j = 0
#
#         while (j <= window):
#             s = s + sound_square[i + j]
#             j = j + 1
#
#         xsum.append(s)
#         c = c + 1
#         count += s
#         i = i + window
#
#     i = 0
#     fx = 0
#     avg = count / c
#     threshold = avg / 30.00
#
#     for i in range(len(xsum)):
#         if xsum[i] > threshold and fx == 0:
#             fx = 1
#             start.append(i * window)
#         elif xsum[i] < threshold and fx == 1:  # end of the sound
#             end.append(i * window)
#             fx = 0
#
#         else:
#             continue
#
#     if len(start) != len(end):
#         end.append(i * window)
#
#     for z in range(len(start)):
#         sx = start[z] / 44100.00
#         Onsets.append(round(sx, 2))
#
#     i = 0
#     while (i < len(start)):
#         dft = np.array(np.fft.fft(sound[start[i]:end[i]]))
#         indexes, _ = scipy.signal.find_peaks(dft, height=45, distance=45)
#         i_max = indexes[0]
#         fr = ((i_max) * sampling_freq) / (end[i] - start[i])
#         idx = (np.abs(array - fr)).argmin()
#         Detected_Notes.append(notes[idx])
#         i = i + 1
#     return Onsets, Detected_Notes
#
#
# ############################### Main Function #############################################
#
# if __name__ == "__main__":
#
#     #   Instructions
#     #   ------------
#     #   Do not edit this function.
#
#     # code for checking output for single audio file
#     # path = os.getcwd()
#
#     # file_name = "C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AndreHolzapfel/bongo1.wav"
#     file_name = "plik.wav"
#     audio_file = wave.open(file_name)
#
#
#     Onsets, Detected_Notes = onset_detect(audio_file)
#
#     print("\n\tOnsets = " + str(Onsets))
#     print("\n\tDetected Notes = " + str(Detected_Notes))
#
#     # code for checking output for all audio files
#     # x = raw_input("\n\tWant to check output for all Audio Files - Y/N: ")
#
#     # if x == 'Y':
#
#     Onsets_list = []
#     Detected_Notes_list = []
#
#     # file_count = len(os.listdir(path + "\Task_1.2A_Audio_files"))
#
#     # for file_number in range(1, file_count):
#     #     file_name = path + "\Task_1.2A_Audio_files\Audio_" + str(file_number) + ".wav"
#     audio_file = wave.open(file_name)
#
#     Onsets, Detected_Notes = onset_detect(audio_file)
#
#     Onsets_list.append(Onsets)
#     Detected_Notes_list.append(Detected_Notes)
#
#     print("\n\tOnsets = " + str(Onsets_list))
#     print("\n\tDetected Notes = " + str(Detected_Notes_list))
#
# # import librosa
# # import sqlite3
# # import pyaudio
# # import numpy as np
# # import wave
# # import time
# # import contextlib
# #
# # import sqlite3
# # import shutil
# # import os
# #
# # # Utwórz bazę danych SQLite
# # conn = sqlite3.connect('audio_database.db')
# # cursor = conn.cursor()
# #
# # # Utwórz tabelę dla plików WAV
# # cursor.execute('''
# #     CREATE TABLE IF NOT EXISTS audio_files (
# #         id INTEGER PRIMARY KEY,
# #         filename TEXT,
# #         data BLOB
# #     )
# # ''')
# #
# # # Ścieżka do katalogu z plikami WAV
# # wav_directory = "C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AlexandreLacoste"
# #
# #
# # # Function to insert a WAV file into the database
# # def insert_wav_file(filename):
# #     wav_name = os.path.basename(filename)  # Get just the file name
# #     with open(filename, 'rb') as wav_file:
# #         wav_data = wav_file.read()
# #         cursor.execute('INSERT INTO audio_files (filename, data) VALUES (?, ?)', (wav_name, wav_data))
# #         conn.commit()
# #
# #
# # # Traverse the directory and insert WAV files into the database
# # for root, _, files in os.walk(wav_directory):
# #     for file in files:
# #         if file.endswith('.wav'):
# #             file_path = os.path.join(root, file)
# #             insert_wav_file(file_path)
# #
# # # Close the database connection
# # conn.close()
# #
# #
# # # try:
# # #     cur
# # # Constants
# #
# # # # check duration of the file:
# # # fname = 'plik.wav'
# # # with contextlib.closing(wave.open(fname, 'r')) as f:
# # #     frames = f.getnframes()
# # #     rate = f.getframerate()
# # #     duration = frames / float(rate)
# # #     print(duration)
# #
# #
# # def sound_detection(threshold=40, duration=5):
# #     CHUNK = 1024  # Size of each audio chunk
# #     FORMAT = pyaudio.paInt16
# #     CHANNELS = 1
# #     RATE = 44100  # Sampling rate (samples per second)
# #
# #     p = pyaudio.PyAudio()
# #     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
# #
# #     sound_detected = False
# #     timestamps = []
# #     start_time = time.time()
# #     print("I am waiting for a sound")
# #
# #     try:
# #         while time.time() - start_time < duration:
# #             data = stream.read(CHUNK)
# #             audio_data = np.frombuffer(data, dtype=np.int16)
# #             amplitude = np.max(audio_data)
# #
# #             if amplitude > threshold:
# #                 if not sound_detected:
# #                     sound_detected = True
# #                     print("sound detected")
# #                     timestamps.append(time.time() - start_time)
# #             else:
# #                 sound_detected = False
# #
# #     except KeyboardInterrupt:
# #         pass
# #
# #     stream.stop_stream()
# #     stream.close()
# #     p.terminate()
# #
# #     with open('sound_timestamps.txt', 'w') as f:
# #         for timestamp in timestamps:
# #             f.write(f'Sound detected at {timestamp}\n')
# #
# #     # for i, timestamp in enumerate(timestamps):
# #     #     print(f"Sound {i + 1} detected at {timestamp:.2f} seconds")
# #
# #
# # # sound_detection()
# # def detect_and_save_onsets(audio_file, output_file=None):
# #     # Load the audio file
# #     y, sr = librosa.load(audio_file)
# #
# #     # Detect onsets
# #     onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
# #     onset_times = librosa.frames_to_time(onset_frames, sr=sr)
# #
# #     # Keep the first 16 onsets
# #     first_16_onsets = onset_times[:16]
# #
# #     # if output_file:
# #     #     with open(output_file, 'w') as f:
# #     #         for onset_time in first_16_onsets:
# #     #             f.write(str(onset_time) + '\n')
# #     # else:
# #     with open('songs.txt', 'w') as f:
# #         for timestamp in first_16_onsets:
# #             f.write(f'Sound detected at {timestamp}\n')
# #
# #     return first_16_onsets
#
#
# # sound_detection()
# # detect_and_save_onsets("plik.wav")
#
# # CHUNK = 1024  # Size of each audio chunk
# # FORMAT = pyaudio.paInt16
# # CHANNELS = 1
# # RATE = 44100  # Sampling rate (samples per second)
# # THRESHOLD = 1000  # Adjust this threshold to suit your specific sound
# #
# # # Initialize the audio stream
# # p = pyaudio.PyAudio()
# # stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
# #
# # # Initialize variables
# # sound_detected = False
# # timestamps = []
# #
# # try:
# #     while True:
# #         data = stream.read(CHUNK)
# #         audio_data = np.frombuffer(data, dtype=np.int16)
# #         amplitude = np.max(audio_data)
# #
# #         if amplitude > THRESHOLD:
# #             if not sound_detected:
# #                 sound_detected = True
# #                 timestamps.append(time.time())
# #         else:
# #             sound_detected = False
# #
# # except KeyboardInterrupt:
# #     pass
# # # Close the audio stream
# # stream.stop_stream()
# # stream.close()
# # p.terminate()
# #
# # # Write timestamps to a file
# # with open('sound_timestamps.txt', 'w') as f:
# #     for timestamp in timestamps:
# #         f.write(f'Sound detected at {timestamp}\n')
#
#
# # def sound_detector():
# #     pa = pyaudio.PyAudio()
# #     sample_rate = 44100
# #     frames_per_buffer = 1024
# #
# #     stream = pa.open(format=pyaudio.paInt16,
# #                      channels=1,
# #                      rate=sample_rate,
# #                      frames_per_buffer=frames_per_buffer,
# #                      input=True)
# #
# #     threshold = 9200
# #     isQuiet = False
# #     try:
# #         while True:
# #             data = stream.read(frames_per_buffer)
# #
# #             # Convert the binary data to a list of integers
# #             data_int = [int(byte) for byte in data]
# #
# #             # Calculate the RMS (Root Mean Square)
# #             rms = sum(x ** 2 for x in data_int) / len(data_int)
# #             oldIsQuiet = isQuiet
# #             isQuiet = rms < threshold
# #             if isQuiet != oldIsQuiet:
# #                 print("it's quiet" if isQuiet else "it's loud")
# #
# #     except KeyboardInterrupt:
# #         pass  # Allow for a clean exit when Ctrl+C is pressed
# #
# #     finally:
# #         stream.stop_stream()
# #         stream.close()
# #         pa.terminate()
