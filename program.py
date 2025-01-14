import wave
import pyaudio
import librosa
import os
from dtw import *
from onset_strength import *
from edit_distance import *
from audio_recording import *

# plot_onset_strength()
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


def compare_with_database(user_audio, database, show_user=False, show_database=False, show_cost_function=False):
    user_audio1, user_sr1 = load_audio(user_audio)
    if show_user:
        user_onset_strength = calculate_onset_strength(user_audio1[44100:], user_sr1, True)
        plot_onset_strength(user_onset_strength, user_audio)
    else:
        user_onset_strength = calculate_onset_strength(user_audio1[44100:], user_sr1, True)
    detected_user_onsets = detect_onsets_dynamic_threshold(user_onset_strength)
    # plot_onset_strength(user_onset_strength, "User Audio")
    distances = []
    for db_entry in database:
        db_file_path = db_entry['file_path']
        db_audio, db_sr = load_audio(db_file_path)
        db_file_name = os.path.basename(db_file_path)

        onset_strength = calculate_onset_strength(db_audio, db_sr)

        if show_database:
            plot_onset_strength(onset_strength, db_file_name)

        db_onsets = detect_onsets_dynamic_threshold(onset_strength)

        # calculate_edit_distance4f, p4, md4 = calculate_edit_distance4(detected_user_onsets, db_onsets, 1.8,
        #                                                               db_file_name)
        # distance_calculate_edit_distance, path = calculate_edit_distance(detected_user_onsets, db_onsets, 2.5,
        #                                                                  db_file_name)  # tu trzeba pokombinować z tym Tm
        # # własne, nowe, odległość edycyjna
        # distance_calculate_dtw_librosa_onsets_edit_distance, path = calculate_dtw_librosa_onsets_edit_distance(
        #     detected_user_onsets, db_onsets, db_file_name)
        # # second version of detecting onset, binary (where the onset is then: 1) - distance +/-
        # distance_calculate_dtw_librosa_onsets2, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name)
        calculate_dtw_librosa_onsets_próbkowanief, path = calculate_dtw_librosa_onsets_sampling(detected_user_onsets, db_onsets, db_file_name)
        # distance, path = calculate_dtw_librosa(detected_user_onsets, db_onsets, db_file_name)
        # calculate_dtwf = calculate_dtw(detected_user_onsets, db_onsets)
        # mean_distance = calculate_edit_distance2(detected_user_onsets,db_onsets, 20, db_file_name)
        distances.append((db_file_name, calculate_dtw_librosa_onsets_próbkowanief))

        # plot_onset_strength(onset_strength, db_file_name)

        if show_cost_function:
            # distance, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name, True, False) # dtw - źle
            # distance, path = calculate_dtw_librosa(user_onset_strength, onset_strength, db_file_name, True) # dtw - ok
            # distance, path = calculate_dtw_librosa_onsets_edit_distance(detected_user_onsets, db_onsets, db_file_name,
            #                                                             True, False) # źle pokazuje dtw
            distance, path = calculate_dtw_librosa_onsets_sampling(detected_user_onsets, db_onsets, db_file_name,
                                                                        True, False)
            # distance, path = calculate_edit_distance(detected_user_onsets, db_onsets, 2.5, db_file_name, True)
            # calculate_edit_distance2(detected_user_onsets, db_onsets, 2, db_file_name, True)

        # print(f'Distance from calculate_edit_distance4 between {user_audio} and {db_file_name}: {calculate_edit_distance4f}')#, mean distance: {mean_distance}')
        # print(f'Distance from calculate_edit_distance between {user_audio} and {db_file_name}: {distance_calculate_edit_distance}')  # , mean distance: {mean_distance}')
        # print(f'Distance from calculate_dtw_librosa_onsets_edit_distance between {user_audio} and {db_file_name}: {distance_calculate_dtw_librosa_onsets_edit_distance}')#, mean distance: {mean_distance}')
        # print(f'Distance from calculate_dtw_librosa_onsets2 between {user_audio} and {db_file_name}: {distance_calculate_dtw_librosa_onsets2}')#, mean distance: {mean_distance}')
        # print(f'Distance from calculate_dtw_librosa_onsets_próbkowanie between {user_audio} and {db_file_name}: {calculate_dtw_librosa_onsets_próbkowanief}')
        # print(f'Distance from calculate_dtw_librosa between {user_audio} and {db_file_name}: {distance}')  # , mean distance: {mean_distance}')
        # print(f'Distance from calculate_dtw (fastdtw) between {user_audio} and {db_file_name}: {calculate_dtwf}')
        # print(f"Średnia odległość punktów na ścieżce od prostej between {user_audio} and {db_file_name}: {mean_distance}")
        # średnia odległość punktów od prostej
        # calculate_edit_distance2(detected_user_onsets,db_onsets, 2, db_file_name)

    # top1_song = min(distances, key=lambda x: x[1])
    # top1_song = max(distances, key=lambda x: x[1])
    # print(f'\nTop 1 song is: {top1_song[0]} with distance: {top1_song[1]}')

    sorted_distances = sorted(distances, key=lambda x: x[1])
    print("\nTop 5 songs:")
    for i, (song, distance) in enumerate(sorted_distances[:5], 1):
        print(f'{i}. {song} with distance: {distance}')

    # sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
    # print("\nTop 3 songs with the greatest distance:")
    # for i, (song, distance) in enumerate(sorted_distances[:5], 1):
    #     print(f'{i}. {song} with distance: {distance}')


def load_database(folder_path):
    audio_files = get_audio_files_in_folder(folder_path)
    database_files = [{'file_path': file} for file in audio_files]
    return database_files


def compare_with_database_ons(user_audio, database, database_onsets_files, show_user=False, show_database=False,
                          show_cost_function=False):
    user_audio1, user_sr1 = load_audio(user_audio)
    if show_user:
        user_onset_strength = calculate_onset_strength(user_audio1[44100:], user_sr1, True)
        plot_onset_strength(user_onset_strength, user_audio)
    else:
        user_onset_strength = calculate_onset_strength(user_audio1[44100:], user_sr1, True)
    detected_user_onsets = detect_onsets_dynamic_threshold(user_onset_strength)
    # detected_user_onsets = load_onsets_mirex(user_audio)
    distances = []
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


        # #z bazą FTIMSET
        # calculate_edit_distance4f, p4, md4 = calculate_edit_distance4(detected_user_onsets, db_onsets, 4,
        #                                                               db_file_name)
        # distance_calculate_edit_distance, path = calculate_edit_distance(detected_user_onsets, db_onsets, 2.5,
        #                                                                  db_file_name)  # tu trzeba pokombinować z tym Tm
                # # własne, nowe, odległość edycyjna
        # distance_calculate_dtw_librosa_onsets_edit_distance, path = calculate_dtw_librosa_onsets_edit_distance(
        #     detected_user_onsets, db_onsets, db_file_name)
                # # second version of detecting onset, binary (where the onset is then: 1) - distance +/-
                # distance_calculate_dtw_librosa_onsets2, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name)
        calculate_dtw_librosa_onsets_próbkowanief, path = calculate_dtw_librosa_onsets_sampling(detected_user_onsets, db_onsets, db_file_name)
        # distance, path = calculate_dtw_librosa(detected_user_onsets, db_onsets, db_file_name)
        # calculate_dtwf = calculate_dtw(detected_user_onsets, db_onsets)

        # distances.append((db_file_name,distance_calculate_dtw_librosa_onsets_edit_distance))
        distances.append((db_file_name, calculate_dtw_librosa_onsets_sampling(detected_user_onsets, db_onsets, db_file_name)))
        if show_cost_function:
            # distance, path = calculate_dtw_librosa_onsets2(detected_user_onsets, db_onsets, db_file_name, True, False) # dtw - źle
            # distance, path = calculate_dtw_librosa(user_onset_strength, onset_strength, db_file_name, True) # dtw - ok
            # distance, path = calculate_dtw_librosa_onsets_edit_distance(detected_user_onsets, db_onsets, db_file_name,
            #                                                             True, False) # źle pokazuje dtw
            calculate_dtw_librosa_onsets_sampling(detected_user_onsets, db_onsets, db_file_name,
                                                                        True, False)
            # distance, path = calculate_edit_distance(detected_user_onsets, db_onsets, 2.5, db_file_name, True)
            # calculate_edit_distance2(detected_user_onsets, db_onsets, 2, db_file_name, True)

        # print(f'Distance from edit_matrix between {user_audio} and {db_file_name}: {distance_edit_matrix}')#, mean distance: {mean_distance}')
        # print(f'Distance from calculate_edit_distance2 (aligment to the line) between {user_audio} and {db_file_name}: {distance_calculate_edit_distance2}')#, mean distance: {mean_distance}')

        # print(f'\nDistance from calculate_edit_distance4 between {user_audio} and {db_file_name}: {calculate_edit_distance4f}')#, mean distance: {mean_distance}')
        # print(f'Distance from calculate_edit_distance between {user_audio} and {db_file_name}: {distance_calculate_edit_distance}')  # , mean distance: {mean_distance}')
        # print(f'Distance from calculate_dtw_librosa_onsets_edit_distance between {user_audio} and {db_file_name}: {distance_calculate_dtw_librosa_onsets_edit_distance}')#, mean distance: {mean_distance}')
        # print(f'Distance from calculate_dtw_librosa_onsets2 between {user_audio} and {db_file_name}: {distance_calculate_dtw_librosa_onsets2}')#, mean distance: {mean_distance}')
        # print(f'Distance from calculate_dtw_librosa_onsets_próbkowanie between {user_audio} and {db_file_name}: {calculate_dtw_librosa_onsets_próbkowanief}')
        # print(f'Distance from calculate_dtw_librosa between {user_audio} and {db_file_name}: {distance}')  # , mean distance: {mean_distance}')
        # print(f'Distance from calculate_dtw (fastdtw) between {user_audio} and {db_file_name}: {calculate_dtwf}')

        # średnia odległość punktów od prostej
        # calculate_edit_distance2(detected_user_onsets,db_onsets, 2, db_file_name)
    sorted_distances = sorted(distances, key=lambda x: x[1])
    print("\nTop 5 songs:")
    for i, (song, distance) in enumerate(sorted_distances[:5], 1):
        print(f'{i}. {song} with distance: {distance}')

    # sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
    # print("\nTop 3 songs with the greatest distance:")
    # for i, (song, distance) in enumerate(sorted_distances[:5], 1):
    #     print(f'{i}. {song} with distance: {distance}')


# Paths to .wav
# database_folder = os.path.join(os.path.dirname(__file__), 'FTIMSET', 'wav', 'Leveau')
# database_folder = os.path.join(os.path.dirname(__file__), 'wszystkie_utwory_FTIMSET', 'wav')
# database_folder = os.path.join(os.path.dirname(__file__), 'Ftimset_48', 'wav')
database_folder = os.path.join(os.path.dirname(__file__), 'Ftimset_48', 'wav')
# database_folder_testowy = os.path.join(os.path.dirname(__file__), 'testowy', 'wav')
# database_folder_testowy = os.path.join(os.path.dirname(__file__), 'wszystkie_utwory_FTIMSET', 'wav')
database_folder_testowy = os.path.join(os.path.dirname(__file__), 'Ftimset_48', 'wav')

# Paths to .ons
database_onsets = os.path.join(os.path.dirname(__file__), 'wszystkie_utwory_FTIMSET', 'ons')
# database_onsets_testowy = os.path.join(os.path.dirname(__file__), 'testowy', 'ons')
# database_onsets_testowy = os.path.join(os.path.dirname(__file__), 'wszystkie_utwory_FTIMSET', 'ons')
database_onsets_testowy = os.path.join(os.path.dirname(__file__), 'Ftimset_48', 'ons')

# .wav files
database = [{'file_path': file} for file in get_audio_files_in_folder(database_folder)]
database_testowy = [{'file_path': file} for file in get_audio_files_in_folder(database_folder_testowy)]


# .ons files
database_onsets_files = [{'file_path': file} for file in get_audio_files_in_folder(database_onsets)]
database_onsets_files_testowy = [{'file_path': file} for file in get_audio_files_in_folder(database_onsets_testowy)]


# compare_with_database_ons('user_sequences/elec.wav', database_testowy, database_onsets_files_testowy, False, False, False)


import argparse


def main():
    parser = argparse.ArgumentParser(description="Audio Comparison Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for the "record" command
    record_parser = subparsers.add_parser("record", help="Record user audio")
    record_parser.add_argument("--output", type=str, required=True, help="Output file for the recorded audio")
    record_parser.add_argument("--duration", type=int, default=10, help="Duration of the recording in seconds (default: 10)")
    # record_parser.add_argument("--device", type=int, help="Device index for the input audio device")

    # Subparser for the "compare" command
    compare_parser = subparsers.add_parser("compare", help="Compare user audio with the database")
    compare_parser.add_argument("--user-audio", type=str, required=True, help="Path to user audio file")
    compare_parser.add_argument("--database-path", type=str, required=True, help="Path to database folder")
    compare_parser.add_argument("--show-user", action="store_true", help="Display onset strength of user audio")
    compare_parser.add_argument("--show-database", action="store_true", help="Display onset strength of database entries")
    compare_parser.add_argument("--show-cost", action="store_true", help="Display cost function during comparison")
    # compare_parser.add_argument("--show-cost-matrix", action="store_true", help="Display cost matrix during comparison")

    # Subparser for the "plot" command
    plot_parser = subparsers.add_parser("plot", help="Plot onset strength or cost matrix")
    plot_parser.add_argument("--type", type=str, choices=["user", "database", "cost"], required=True,
                              help="Type of plot to display")
    plot_parser.add_argument("--file", type=str, required=True, help="Path to the audio file or cost matrix")

    args = parser.parse_args()

    if args.command == "record":
        # Handle recording
        # print(f"Recording audio to {args.output} for {args.duration} seconds")
        record_audio_pyaudio(args.output, duration=args.duration)

    elif args.command == "compare":
        # Handle comparison
        print(f"Comparing {args.user_audio} with database at {args.database_path}")
        database = load_database(args.database_path)
        compare_with_database(args.user_audio, database, show_user=args.show_user, show_database=args.show_database,
                              show_cost_function=args.show_cost)


    elif args.command == "plot":
        # Handle plotting
        if args.type == "user":
            print(f"Plotting onset strength for user audio: {args.file}")
            user_audio, sr = load_audio(args.file)
            onset_strength = calculate_onset_strength(user_audio, sr, True)
            plot_onset_strength(onset_strength, args.file)
        elif args.type == "database":
            print(f"Plotting onset strength for database audio: {args.file}")
            db_audio, sr = load_audio(args.file)
            onset_strength = calculate_onset_strength(db_audio, sr, True)
            plot_onset_strength(onset_strength, args.file)
        elif args.type == "cost":
            print(f"Plotting cost function matrix for: {args.file}")
            cost_matrix = np.load(args.file)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
# import pyaudio
# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
# print("Recording...")
# frames = []
# for _ in range(0, int(44100 / 1024 * 5)):  # nagrywanie przez 5 sekund
#     data = stream.read(1024)
#     frames.append(data)
# stream.stop_stream()
# stream.close()
# p.terminate()
# print("Done recording.")
