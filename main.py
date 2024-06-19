import wave
import pyaudio
import os
from dtw import *
from onset_strength import *
from edit_distance import *
from audio_recording import *


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
            if file.endswith(('.wav', '.mp3', '.ons', '.onset')):
                audio_files.append(os.path.join(root, file))
    return audio_files


def load_onsets_mirex(file_path):
    with open(file_path, 'r') as file:
        line = file.readline().strip()  # Wczytaj linię i usuń białe znaki z początku i końca
        onsets = [float(value) for value in line.split()]  # Podziel linię na wartości i przekonwertuj je na float
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
    # detected_user_onsets = load_onsets_mirex(user_audio)

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

        db_onsets = load_onsets(db_onsets_file_path)  # do mojej bazy danych
        # db_onsets = load_onsets_mirex(db_onsets_file_path) # do bazy mirex
        # first version
        # distance, path = calculate_dtw_librosa_onsets(detected_user_onsets, db_onsets, db_file_name) #distance: -

        # second version of detecting onset, binary (where the onset is then: 1) - distance +/-
        distance_calculate_dtw_librosa_onsets2, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name)

        # binary but with sampling (lenght of the user the same as in the database) - distance +/-
        # distance, path = calculate_dtw_librosa_onsets_próbkowanie(detected_user_onsets, db_onsets, db_file_name)

        distance_calculate_edit_distance, path = calculate_edit_distance(detected_user_onsets, db_onsets, 2.5, db_file_name)  # tu trzeba pokombinować z tym Tm

        # distance = edit_matrix(detected_user_onsets, db_onsets)
        
        distance_edit_matrix = edit_matrix(detected_user_onsets, db_onsets)
        # # print(f"Value of d before traceback: {d}")
        # path = traceback(detected_user_onsets, db_onsets, distance)
        #
        # visualize_edit_distance_path(detected_user_onsets, db_onsets, path)
        distance_calculate_edit_distance2, path, mean_distance = calculate_edit_distance2(detected_user_onsets, db_onsets, 2, db_file_name)
        distance, path = calculate_dtw_librosa(detected_user_onsets, db_onsets,  db_file_name)
        # własne, nowe, odległość edycyjna
        distance_calculate_dtw_librosa_onsets_edit_distance, path = calculate_dtw_librosa_onsets_edit_distance(detected_user_onsets, db_onsets, db_file_name)
        if show_cost_function:
            # distance, path = calculate_dtw_librosa_onsets(detected_user_onsets, db_onsets, db_file_name, True) # dtw - źle
            # distance, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name, True, False) # dtw - źle
            # distance, path = calculate_dtw_librosa(user_onset_strength, onset_strength, db_file_name, True) # dtw - ok
            # distance, path = calculate_dtw_librosa_onsets_edit_distance(detected_user_onsets, db_onsets, db_file_name,
            #                                                             True, False) # źle pokazuje dtw
            # distance, path = calculate_dtw_librosa_onsets_próbkowanie(detected_user_onsets, db_onsets, db_file_name,
            #                                                             True, True)
            # distance, path = calculate_edit_distance(detected_user_onsets, db_onsets, 2.5, db_file_name, True)
            distance, path, mean_distance = calculate_edit_distance2(detected_user_onsets, db_onsets, 2, db_file_name, True)

        print(f'Distance from calculate_dtw_librosa_onsets_edit_distance between {user_audio} and {db_file_name}: {distance}')#, mean distance: {mean_distance}')
        # print(f'Distance from edit_matrix between {user_audio} and {db_file_name}: {distance_edit_matrix}')#, mean distance: {mean_distance}')
        print(f'Distance from calculate_edit_distance2 between {user_audio} and {db_file_name}: {distance_calculate_edit_distance2}')#, mean distance: {mean_distance}')
        print(f'Distance from calculate_dtw_librosa_onsets_edit_distance between {user_audio} and {db_file_name}: {distance_calculate_dtw_librosa_onsets_edit_distance}')#, mean distance: {mean_distance}')
        print(f'Distance from calculate_edit_distance between {user_audio} and {db_file_name}: {distance_calculate_edit_distance}')#, mean distance: {mean_distance}')
        print(f'Distance from calculate_dtw_librosa_onsets2 between {user_audio} and {db_file_name}: {distance_calculate_dtw_librosa_onsets2}')#, mean distance: {mean_distance}')


def compare_with_database_mirex(user_ons, database_onsets_files, show_cost_function=False):
    detected_user_onsets = load_onsets_mirex(user_ons)

    for db_onsets_entry in database_onsets_files:
        db_onsets_file_path = db_onsets_entry['file_path']
        db_file_name = os.path.basename(db_onsets_file_path)

        db_onsets = load_onsets_mirex(db_onsets_file_path)

        # distance, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name)
        # distance, path = calculate_dtw_librosa_onsets_próbkowanie(detected_user_onsets, db_onsets, db_file_name)

        # distance, path = calculate_dtw_librosa_onsets_edit_distance(detected_user_onsets, db_onsets, db_file_name) # nie działa
        distance, path = calculate_edit_distance(detected_user_onsets, db_onsets, 1.8, db_file_name)

        if show_cost_function:
            # distance, path = calculate_dtw_librosa_onsets_edit_distance(detected_user_onsets, db_onsets, db_file_name,
            #                                                             True, True)

            # distance, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name, True, False)
            # distance, path = calculate_dtw_librosa_onsets_próbkowanie(detected_user_onsets, db_onsets, db_file_name, True, False)

            # distance, path = calculate_dtw_librosa_onsets_edit_distance(detected_user_onsets, db_onsets, db_file_name, True, False)

            distance, path = calculate_edit_distance(detected_user_onsets, db_onsets, 1.8,  db_file_name, True)
        print(f'Distance from librosa between {user_ons} and {db_file_name}: {distance}')


# Paths to .wav
database_folder = os.path.join(os.path.dirname(__file__), 'FTIMSET', 'wav', 'Leveau')

# Paths to .ons
database_onsets = os.path.join(os.path.dirname(__file__), 'FTIMSET', 'ons', 'Leveau')

# .wav files
database = [{'file_path': file} for file in get_audio_files_in_folder(database_folder)]

# .ons files
database_onsets_files = [{'file_path': file} for file in get_audio_files_in_folder(database_onsets)]

# mirex ons
# paths
# mirex_ons_folder = os.path.join(os.path.dirname(__file__), 'qbt-extended-onset', 'qbt-extended-onset')
# mirex_ons_folder = r"C:\Users\gosia\OneDrive\Pulpit\qbt-extended-onset\qbt-extended-onset"
mirex_ons_folder = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\qbt-extended-onset\qbt-extended-onset"

# .ons files
mirex_ons = [{'file_path': file} for file in get_audio_files_in_folder(mirex_ons_folder)]

# arguments for compare_with_database:
# user audio,
# database with wav files,
# database with ons files,
# True or False if show user files onset strength,
# True or False if show database files onset strength,
# True or False if show cost function
# compare_with_database('piano1_krótsze.wav', database, database_onsets_files, False, False, False)
# compare_with_database('user_sequences/sax1.wav', database, database_onsets_files)
compare_with_database('user_sequences/clarinet1.wav', database, database_onsets_files, False, False, False)
# compare_with_database('violin.wav', database, database_onsets_files)
# compare_with_database('user3.1.wav', database, database_onsets_files, False, False, True)
# compare_with_database('002_2x_onset.ons', database, database_onsets_files, False, False, False)

# z bazą danych mirex
# compare_with_database_mirex('002_2x_onset.ons', mirex_ons)
# compare_with_database_mirex('002_2x_onset.ons', mirex_ons, True)
# compare_with_database('user1.wav', database, database_onsets_files, False, False, False)
