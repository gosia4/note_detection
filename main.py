import numpy as np
import wave
import matplotlib.pyplot as plt
import pyaudio
from fastdtw import fastdtw
import dtw
import librosa
import os
from dtw import *

# TODO: poprawić nazwy w macierzy kosztów i wykresie dopasowania w wywołaniu oraz nazwy dla siły onsetów dla wykresu,
#  dodać wykres macierzy kosztów i dtw do compare_with_database


# wczytanie pliku od użytkownika
def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


# siła onsetów
def calculate_onset_strength(y, sr, normalize=True):
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    if normalize:
        onset_strength = librosa.util.normalize(onset_strength)
    return onset_strength


def plot_onset_strength(onset_strength, db_file_name):
    plt.figure(figsize=(12, 6))
    plt.plot(librosa.times_like(onset_strength), onset_strength, label='Onset Strength')
    plt.ylabel('Onset Strength')
    plt.xlabel('Time (s)')
    plt.title(db_file_name)
    plt.show()


def record_audio_pyaudio(filename, duration=5, fs=44100):
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


# calculate dtw using fastdtw library
def calculate_dtw(user, db, show=False, normalize=True):
    # Ucięcie sekwencji, żeby miały tę samą długość
    min_len = min(len(user), len(db))
    user = user[:min_len]
    db = db[:min_len]

    #  normalize, if the speed from the user is different or the sequence is shorter
    if normalize:
        user = librosa.util.normalize(user)
        db = librosa.util.normalize(db)
    distance, path = fastdtw(user, db)

    path_array = np.array(path)
    if show:
        plt.figure(figsize=(12, 6))
        plt.plot(path_array[:, 0], path_array[:, 1], label='Alignment Path',
                 color='red')  # 1 kolumna i 2 kolumna czyli dwie sekwencje
        plt.title('DTW Alignment Path')
        plt.xlabel('User Onset Strength Index')
        plt.ylabel('Database Onset Strength Index')
        plt.legend()
        plt.show()

    return distance, path


# normalize the amplitude, so that if the user sequence has different amplitude, it will match better dtw
def calculate_dtw_librosa(user, db, file_name, show=False, normalize=True):
    # if normalize:
    #     user = librosa.util.normalize(user)
    #     db = librosa.util.normalize(db)

    distance, path = librosa.sequence.dtw(user, db)

    if show:
        # używając librosa wykres dopasowania
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(distance, x_axis='frames', y_axis='frames', ax=ax[0])
        ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
        ax[0].plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
        ax[0].legend()
        fig.colorbar(img, ax=ax[0])
        ax[1].plot(distance[-1, :] / path.shape[0])
        ax[1].set(xlim=[0, db.shape[0]], ylim=[0, 2], title='Matching cost function: ' + file_name)
        plt.show()
    return np.mean(distance), path


# spectrum
def calculate_and_plot_spectrum(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # amplituda sygnału do skali decybelowej
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


def get_audio_files_in_folder(folder_path):
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                audio_files.append(os.path.join(root, file))
    return audio_files


database_folder = 'C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/Leveau'
database = [{'file_path': file} for file in get_audio_files_in_folder(database_folder)]


def compare_with_database(user_audio, database, show_user=False, show_database=False, show_cost_function = False):
    user_audio1, user_sr1 = load_audio(user_audio)
    if show_user:
        user_onset_strength = calculate_onset_strength(user_audio1, user_sr1, True)
    else:
        user_onset_strength = calculate_onset_strength(user_audio1, user_sr1, False)
    for entry in database:
        db_file_path = entry['file_path']
        db_audio, db_sr = load_audio(db_file_path)
        db_file_name = os.path.basename(db_file_path)
        # Oblicz siłę onsetów dla każdego pliku w bazie danych
        # db_onset_strength = calculate_onset_strength(db_audio, db_sr, False)
        onset_strength = calculate_onset_strength(db_audio, db_sr)
        if show_database:
            plot_onset_strength(onset_strength, db_file_name)

        # onset_strength = librosa.onset.onset_strength(y=db_audio, sr=db_sr)
        # if show_database:
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(librosa.times_like(onset_strength), onset_strength, label='Onset Strength')
        #     plt.ylabel('Onset Strength')
        #     plt.xlabel('Time (s)')
        #     plt.title(db_file_name)
        #     plt.show()
        distance, path = calculate_dtw_librosa(user_onset_strength, onset_strength, db_file_name, True)

        print(f'Distance from librosa between {user_audio} and {db_file_name}: {distance}')

compare_with_database('user1.wav', database, True, True, True)

# def main():
# Wczytaj plik audio użytkownika
# user_record = record_audio_pyaudio('user3.wav', 15)
# user_file_path = 'user.wav'
# user_audio, user_sr = load_audio('user.wav')
# user_audio1, user_sr1 = load_audio('user1.wav') # zapytanie podobne do piano1
# user_audio2, user_sr2 = load_audio('user2.wav') # zapytanie podobne do piano1
# user_audio3, user_sr3 = load_audio('user3.wav') # do klarnetu
# user_audio4, user_sr4 = load_audio('piano1_krótsze.wav')
# user_audio1, user_sr1 = load_audio('C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/Leveau/piano1.wav')
#
# # Oblicz siłę onsetów
# user_onset_strength = calculate_onset_strength(user_audio[22050:], user_sr, False)
# calculate_onset_strength(user_audio1, user_sr1, False)
# user_onset_strength1 = calculate_onset_strength(user_audio1[44100:], user_sr1, False)
# user_onset_strength2 = calculate_onset_strength(user_audio2[22050:], user_sr2, False)
# user_onset_strength3 = calculate_onset_strength(user_audio3[22050:], user_sr3, False)
# user_onset_strength4 = calculate_onset_strength(user_audio4, user_sr4, False)

# Oblicz i wyświetl spectrum dla audio użytkownika
# calculate_and_plot_spectrum(user_audio3[22050:], user_sr3)

# Baza danych plików audio
# database = [
#     {'file_path': 'C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/Pertusa/tiersen11.wav'},
#     {'file_path': 'C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/AndreHolzapfel/bongo1.wav'},
#     {'file_path': 'C:/Users/gosia/OneDrive/Pulpit/pythonProject/plik.wav'}
# ]
# database_folder = 'C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/Leveau'
# database = [{'file_path': file} for file in get_audio_files_in_folder(database_folder)]
# #
# #
# for entry in database:
#     db_file_path = entry['file_path']
#     db_audio, db_sr = load_audio(db_file_path)
#     db_file_name = os.path.basename(db_file_path)
#     # Oblicz siłę onsetów dla każdego pliku w bazie danych
#     db_onset_strength = calculate_onset_strength(db_audio, db_sr, True)
    # onset_strength = librosa.onset.onset_strength(y=db_audio, sr=db_sr)
    # plt.figure(figsize=(12, 6))
    # plt.plot(librosa.times_like(onset_strength), onset_strength, label='Onset Strength')
    # plt.ylabel('Onset Strength')
    # plt.xlabel('Time (s)')
    # plt.title(db_file_name)
    # plt.show()


    # normalizacja tempa
    # user_onset_strength_normalized = librosa.effects.preemphasis(user_onset_strength4)
    # db_onset_strength_normalized = librosa.effects.preemphasis(db_onset_strength)
    # tempo_ratio = len(db_audio) / len(user_audio4)
    # user_audio_normalized_tempo = librosa.effects.time_stretch(user_audio4, rate=tempo_ratio)
    # używając time_strech
    # user_onset_strength_normalized = librosa.effects.time_stretch(user_onset_strength4,
    #                                                               rate=(len(db_onset_strength) / len(
    #                                                                   user_onset_strength4)))

    # distance4, path = match_user_to_database(user_onset_strength4, db_onset_strength)
    # distance, path = calculate_dtw_librosa(user_onset_strength1, db_onset_strength, True)
    # distance4, path4 = calculate_dtw_librosa(user_onset_strength4, db_onset_strength)
    # z normalizacją tempa
    # distance4, path = calculate_dtw_librosa(user_onset_strength_normalized, db_onset_strength)

    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # img = librosa.display.specshow(distance, x_axis='frames', y_axis='frames', ax=ax[0])
    # ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
    # ax[0].plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
    # ax[0].legend()
    # fig.colorbar(img, ax=ax[0])
    # ax[1].plot(distance[-1, :] / path.shape[0])
    # ax[1].set(xlim=[0, db_onset_strength.shape[0]], ylim=[0, 2], title='Matching cost function')
    # plt.show()

    # distance, path1 = librosa.sequence.dtw(user_onset_strength1, onset_strength)  # tu podać z zapytania
    # distance2, path2 = librosa.sequence.dtw(user_onset_strength2, db_onset_strength)  # tu podać z zapytania
    # distance4, path4 = librosa.sequence.dtw(user_onset_strength4, db_onset_strength)  # tu podać z zapytania
    # average_distance1 = np.mean(distance)
    # average_distance2 = np.mean(distance2)
    # average_distance2 = np.mean(distance4)
    # print(f'Distance from librosa between user1.wav and {db_file_name}: {distance}')
    # print(f'Distance from librosa between piano1krótsze.wav and {db_file_name}: {distance4}')

    # distancefast, path1fast = fastdtw(user_onset_strength1, db_onset_strength)  # tu podać z zapytania
    # distancefast2, path2fast = fastdtw(user_onset_strength2, db_onset_strength)  # tu podać z zapytania
    # distancefast2, path2fast = fastdtw(user_onset_strength4, db_onset_strength)  # tu podać z zapytania
    # print(f'Distance from fastdtw between user1.wav and {db_file_name}: {distancefast}')
    # print(f'Distance from fastdtw between user1.wav and {db_file_name}: {distancefast2}\n')

# calculate_onset_strength(user_audio1[44100:], user_sr1, True)
# user_onset_strength2 = calculate_onset_strength(user_audio2[22050:], user_sr2, False)
# user_onset_strength3 = calculate_onset_strength(user_audio3[22050:], user_sr3, False)
# calculate_onset_strength(user_audio4, user_sr4, True)

# print(f'The best match sequence is {best_match_name}')

# db_audio, db_sr = load_audio('C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/Leveau/piano1.wav')
# # db_audio, db_sr = load_audio('C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/Leveau/clarinet1.wav')
# db_onset_strength = calculate_onset_strength(db_audio, db_sr, False)
#
# distance1, path1 = librosa.sequence.dtw(user_onset_strength1, db_onset_strength, subseq=True)
# distance2, path2 = librosa.sequence.dtw(user_onset_strength2, db_onset_strength, subseq=True)
# # distance3, path3 = librosa.sequence.dtw(user_onset_strength4, db_onset_strength, subseq=True)
# distance4, path4 = librosa.sequence.dtw(user_onset_strength4, db_onset_strength, subseq=True)
# average_distance = np.mean(distance1)
# average_distance2 = np.mean(distance2)
# # average_distance3 = np.mean(distance3)
# average_distance4 = np.mean(distance4)
# print(f'Distance between user1 and piano1: {average_distance}\n')
# print(f'Distance between user2 and piano1: {average_distance2}\n')
# print(f'Distance between piano1krótsze and piano1: {average_distance4}\n')
# # print(f'Distance between user3 and clarinet1: {average_distance3}\n')
#

# używając numpy
# distance = dtw(user_onset_strength, db_onset_strength)
# distance.plot()
# print(f'Distance between user.wav and od c do g: {distance}\n')

# używając librosa wykres dopasowania
# fig, ax = plt.subplots(nrows=2, sharex=True)
# img = librosa.display.specshow(distance3, x_axis='frames', y_axis='frames', ax=ax[0])
# ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
# ax[0].plot(path3[:, 1], path3[:, 0], label='Optimal path', color='y')
# ax[0].legend()
# fig.colorbar(img, ax=ax[0])
# ax[1].plot(distance3[-1, :] / path3.shape[0])
# ax[1].set(xlim=[0, db_onset_strength.shape[0]], ylim=[0, 2], title='Matching cost function')
# plt.show()
# fig, ax = plt.subplots(nrows=2, sharex=True)
# img = librosa.display.specshow(distance2, x_axis='frames', y_axis='frames', ax=ax[0])
# ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
# ax[0].plot(path2[:, 1], path2[:, 0], label='Optimal path', color='y')
# ax[0].legend()
# fig.colorbar(img, ax=ax[0])
# ax[1].plot(distance1[-1, :] / path2.shape[0])
# ax[1].set(xlim=[0, db_onset_strength.shape[0]], ylim=[0, 2], title='Matching cost function')
# plt.show()

# path: dopasowanie do konkretnych onsetów,
#  czyli (1, 1) to oznacza najlepsze dopaswoanie 1 dźwięku do 1 dźwięku,
#  a np. (8,2) oznacza najlepsze dopasowanie 8 dźwięku do 2 dźwięku


# wykres dopasowania pathu z fastdtw
# librosa.sequence.dtw sprawdzić
# po sekwensji
# zsumować i podzielić przez liczbe elem, z path żeby uzyskać distance
# return steps, ścieżka w formie próbek
# na innych nahraniach
# pianino z leveau
# zbadać jak zachowuje się wyszukiwanie
# przygoować kilka zapyań żeby uzyskać wyniki dla wszystkich zapytań dla całej bazy



# normalizować wartość onsetów, najwyższy = 1
# wyświetlić macierz kosztów dla każdego nagrania z zapytaniem














def match_user_to_database(user_onset_strength, database_onset_strength):
    user_len = len(user_onset_strength)
    db_len = len(database_onset_strength)

    min_distance = float('inf')
    best_match_index = 0
    best_match_name = ""

    for i in range(db_len - user_len + 1):
        # if i + user_len <= len(database):
        db_slice = database_onset_strength[i:i + user_len]
        distance, _ = calculate_dtw_librosa(user_onset_strength, db_slice)

        if distance < min_distance:
            min_distance = distance
            best_match_index = i
            # best_match_name = database[i]['file_path']

    return min_distance, best_match_index


def match_user_to_database_fastdtw(user_onset_strength, database_onset_strength):
    user_len = len(user_onset_strength)
    db_len = len(database_onset_strength)

    min_distance = float('inf')
    best_match_index = 0

    for i in range(db_len - user_len + 1):
        db_slice = database_onset_strength[i:i + user_len]
        distance, _ = calculate_dtw(user_onset_strength, db_slice, normalize=True)

        if distance < min_distance:
            min_distance = distance
            best_match_index = i

    return min_distance, best_match_index


# dla fastdtw przyklad

user_audio1, user_sr1 = load_audio('user1.wav') # zapytanie podobne do piano1
user_onset_strength1 = calculate_onset_strength(user_audio1[44100:], user_sr1)
db_audio, db_sr = load_audio('C:/Users/gosia/OneDrive/Pulpit/FTIMSET/wav/Leveau/piano1.wav')
db_onset_strength = calculate_onset_strength(db_audio, db_sr, False)
distance = calculate_dtw(user_onset_strength1, db_onset_strength, True)
print(f'Distance fastdtw between user2.wav and piano1: {distance}\n')