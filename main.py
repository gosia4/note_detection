import numpy as np
import wave
import matplotlib.pyplot as plt
import pyaudio
from fastdtw import fastdtw
import dtw
import librosa
import os
from dtw import *


def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


# siła onsetów, normalizacja od 0 do 1
def calculate_onset_strength(y, sr, normalize=True):
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    if normalize:
        onset_strength = librosa.util.normalize(onset_strength)
    return onset_strength


def plot_onset_strength(onset_strength, db_file_name, save=False, name='name'):
    plt.figure(figsize=(12, 6))
    plt.plot(librosa.times_like(onset_strength), onset_strength, label='Onset Strength')
    plt.ylabel('Onset Strength')
    plt.xlabel('Time (s)')
    plt.title(db_file_name)
    if save:
        plt.savefig(name + '.png')
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


def detect_onsets_dynamic_threshold(onset_strength, threshold_factor=0.02, fs=44100):
    onsets = []  # List with times of the onsets

    # Average and standard deviation of the onsets
    average_strength = np.mean(onset_strength)
    std_strength = np.std(onset_strength)

    # DYnamic treshlod - arythmetic mean and standard deviation
    threshold = average_strength + threshold_factor * std_strength

    # Onset detection
    for i in range(1, len(onset_strength)):
        if onset_strength[i] > threshold >= onset_strength[i - 1]:
            onsets.append(librosa.frames_to_time(i, sr=fs))

    return onsets


# dtw for onset strength
def calculate_dtw_librosa(user, db, file_name, show=False, save=False, normalize=True):
    # normalize the amplitude, so that if the user sequence has different amplitude, it will match better dtw
    # if normalize:
    #     user = librosa.util.normalize(user)
    #     db = librosa.util.normalize(db)

    distance, path = librosa.sequence.dtw(user, db)

    if show:
        # from librosa
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(distance, x_axis='frames', y_axis='frames', ax=ax[0])
        ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
        ax[0].plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
        ax[0].legend()
        fig.colorbar(img, ax=ax[0])
        ax[1].plot(distance[-1, :] / path.shape[0])
        ax[1].set(xlim=[0, db.shape[0]], ylim=[0, 2], title='Matching cost function: ' + file_name)
        if save:
            plt.savefig('dtw' + file_name + '.png')
        plt.show()
    return np.mean(distance), path


# user - onset strength of the user file
# db - onset strength of the database file
# directly on user files and db files
def calculate_dtw_librosa_onsets(user, db, file_name, show=False, save=False):
    distance, path = librosa.sequence.dtw(user, db)

    if show:
        # from librosa
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(distance, x_axis='frames', y_axis='frames', ax=ax[0])
        ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
        ax[0].plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
        ax[0].legend()
        fig.colorbar(img, ax=ax[0])
        ax[1].plot(distance[-1, :] / path.shape[0])
        # ax[1].set(xlim=[0, db.shape[0]], ylim=[0, 2], title='Matching cost function: ' + file_name)
        ax[1].set(xlim=[0, len(db)], ylim=[0, 2], title='Matching cost function: ' + file_name)

        if save:
            plt.savefig('dtw' + file_name + '.png')
        plt.show()
    return np.mean(distance), path


# better distance as an output than from calculate_dtw_librosa_onsets, but the cost matrix is not understandable
# transpose to binary system, where each 1 means an onset
def calculate_dtw_librosa_onsets2(user, db, file_name, show=False, save=False):
    user_sequence = np.zeros(len(user))
    db_sequence = np.zeros(len(user))  # length of the db files the same as the user files
    for user_onset in user:
        user_sequence[int(user_onset)] = 1
    for db_onset in db:
        if int(db_onset) < len(db_sequence):  # Checking if the index is in range
            db_sequence[int(db_onset)] = 1

    distance, path = librosa.sequence.dtw(user_sequence, db_sequence)

    if show:
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(distance, x_axis='frames', y_axis='frames', ax=ax[0])
        ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
        ax[0].plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
        ax[0].legend()
        fig.colorbar(img, ax=ax[0])
        ax[1].plot(distance[-1, :] / path.shape[0])
        # ax[1].set(xlim=[0, db.shape[0]], ylim=[0, 2], title='Matching cost function: ' + file_name)
        ax[1].set(xlim=[0, len(db)], ylim=[0, 2], title='Matching cost function: ' + file_name)
        if save:
            plt.savefig('dtw' + file_name + '.png')
        plt.show()

    average_distance = np.mean(distance)

    return average_distance, path


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
            if file.endswith(('.wav', '.mp3', '.ons')):
                audio_files.append(os.path.join(root, file))
    return audio_files


# for .ons files
def load_onsets(file_path):
    with open(file_path, 'r') as file:
        onsets = [float(line.strip()) for line in file]
    return onsets


def compare_with_database(user_audio, database, database_onsets_files, show_user=False, show_database=False,
                          show_cost_function=False):
    user_audio1, user_sr1 = load_audio(user_audio)
    if show_user:
        user_onset_strength = calculate_onset_strength(user_audio1[44100:], user_sr1, True)
        plot_onset_strength(user_onset_strength, user_audio)
    else:
        user_onset_strength = calculate_onset_strength(user_audio1[44100:], user_sr1, True)
    detected_user_onsets = detect_onsets_dynamic_threshold(user_onset_strength)

    for db_entry, db_onsets_entry in zip(database, database_onsets_files):
        db_file_path = db_entry['file_path']
        db_onsets_file_path = db_onsets_entry['file_path']

        db_audio, db_sr = load_audio(db_file_path)
        db_file_name = os.path.basename(db_file_path)

        onset_strength = calculate_onset_strength(db_audio, db_sr)

        if show_database:
            plot_onset_strength(onset_strength, db_file_name)

        # without detecting onset, just onset strength
        # distance, path = calculate_dtw_librosa(user_onset_strength, onset_strength, db_file_name, False)

        db_onsets = load_onsets(db_onsets_file_path)
        # first version
        # distance, path = calculate_dtw_librosa_onsets(detected_user_onsets, db_onsets, db_file_name)

        # second version of detecting onset, binary (where the onset is then - 1)
        distance, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name)
        if show_cost_function:
            # distance, path = calculate_dtw_librosa_onsets(detected_user_onsets, db_onsets, db_file_name, True, True)
            distance, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name, True, True)
            # distance, path = calculate_dtw_librosa(user_onset_strength, onset_strength, db_file_name, True)

        print(f'Distance from librosa between {user_audio} and {db_file_name}: {distance}')


# Paths to .wav
database_folder = os.path.join(os.path.dirname(__file__), 'FTIMSET', 'wav', 'Leveau')

# Paths to .ons
database_onsets = os.path.join(os.path.dirname(__file__), 'FTIMSET', 'ons', 'Leveau')

# .wav files
database = [{'file_path': file} for file in get_audio_files_in_folder(database_folder)]

# .ons files
database_onsets_files = [{'file_path': file} for file in get_audio_files_in_folder(database_onsets)]


# arguments for compare_with_database:
# user audio,
# database with wav files,
# database with ons files,
# True or False if show user files onset strength,
# True or False if show database files onset strength,
# True or False if show cost function
compare_with_database('user3.wav', database, database_onsets_files, True, True, True)
# compare_with_database('user1.wav', database, database_onsets_files, False, False, False)

