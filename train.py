import os
import librosa
from onset_strength import *
from edit_distance import *
from audio_recording import *
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# Funkcja do wczytania i przetworzenia sekwencji dźwiękowych użytkownika z folderu
def process_user_sequences_from_folder(folder_path):
    user_sequences = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            audio_file_path = os.path.join(folder_path, file_name)
            y, sr = librosa.load(audio_file_path)
            # Tutaj możesz przeprowadzić dodatkowe operacje przetwarzania, jeśli są potrzebne
            user_sequences.append(y)
    return user_sequences

# Funkcja do wczytania sekwencji dźwiękowych z bazy danych
def load_db_sequences_from_folder(folder_path):
    db_sequences = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            audio_file_path = os.path.join(folder_path, file_name)
            y, sr = librosa.load(audio_file_path)
            # Tutaj możesz przeprowadzić dodatkowe operacje przetwarzania, jeśli są potrzebne
            db_sequences.append(y)
    return db_sequences

# Funkcja do obliczania siły onsetów
def calculate_onset_strength(y, sr, normalize=True):
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    if normalize:
        onset_strength = librosa.util.normalize(onset_strength)
    return onset_strength


def compare_with_database(user_audio, database, database_onsets_files, show_user=False, show_database=False,
                          show_cost_function=False):
    user_audio1, user_sr1 = load_audio(user_audio)
    if show_user:
        user_onset_strength = calculate_onset_strength(user_audio1[44100:], user_sr1, True)
        plot_onset_strength(user_onset_strength, user_audio)
    else:
        user_onset_strength = calculate_onset_strength(user_audio1[44100:], user_sr1, True)
    detected_user_onsets = detect_onsets_dynamic_threshold(user_onset_strength)

    labels_edit_distance = []
    labels_alignment_to_line = []

    for db_entry, db_onsets_entry in zip(database, database_onsets_files):
        db_file_path = db_entry['file_path']
        db_onsets_file_path = db_onsets_entry['file_path']

        db_audio, db_sr = load_audio(db_file_path)
        db_file_name = os.path.basename(db_file_path)

        onset_strength = calculate_onset_strength(db_audio, db_sr)

        if show_database:
            plot_onset_strength(onset_strength, db_file_name)

        db_onsets = load_onsets(db_onsets_file_path)

        distance, path, mean_distance = calculate_edit_distance2(detected_user_onsets, db_onsets, 3, db_file_name)

        if show_cost_function:
            distance, path, mean_distance = calculate_edit_distance2(detected_user_onsets, db_onsets, 2, db_file_name,
                                                                     True)

        labels_edit_distance.append(distance)
        labels_alignment_to_line.append(mean_distance)

    return labels_edit_distance, labels_alignment_to_line


# Ścieżka do folderu z sekwencjami użytkownika
user_folder_path = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\note_detection\user_sequences"

# Ścieżka do folderu z sekwencjami w bazie danych
db_folder_path = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\note_detection\FTIMSET\wav\Leveau"

# Wczytanie i przetworzenie sekwencji dźwiękowych użytkownika z folderu
user_sequences = process_user_sequences_from_folder(user_folder_path)

# Wczytanie sekwencji dźwiękowych z bazy danych z folderu
db_sequences = load_db_sequences_from_folder(db_folder_path)

# Normalizacja danych wejściowych
scaler = MinMaxScaler()
user_sequences_scaled = scaler.fit_transform(user_sequences)
db_sequences_scaled = scaler.transform(db_sequences)

# Etykiety: odległość edycyjna i ocena dopasowania do prostej
# labels_edit_distance = [...]  # Lista odległości edycyjnych dla każdej sekwencji w bazie danych
# labels_alignment_to_line = [...]  # Lista ocen dopasowania do prostej dla każdej sekwencji w bazie danych
# Lista z dopasowaniami między sekwencjami użytkownika a sekwencjami z bazy danych
matches = [
    ("1 i 2 sekwencja użytkownika", "clarinet.1.wav"),
    ("3 sekwencja użytkownika", "od c do g"),
    ("4, 5 i 6 sekwencja użytkownika", "piano1.wav"),
    ("7 sekwencja użytkownika", "sax1.wav")
]

# Odpowiadające im etykiety
labels = [
    "clarinet",
    "od c do g",
    "piano",
    "sax"
]

# Etykiety: odległość edycyjna i ocena dopasowania do prostej
labels_edit_distance = []
labels_alignment_to_line = []

# Dodatkowe etykiety dla dopasowań
additional_labels_train = []

# Przetwarzanie dopasowań na etykiety
for user_seq, db_seq in matches:
    if "clarinet1" in db_seq:
        labels_edit_distance.append(0)  # Dla clarinet oznaczam jako 0
    elif "od c do g" in db_seq:
        labels_edit_distance.append(1)  # Dla od c do g oznaczam jako 1
    elif "piano1" in db_seq:
        labels_edit_distance.append(2)  # Dla piano oznaczam jako 2
    elif "sax1" in db_seq:
        labels_edit_distance.append(3)  # Dla sax oznaczam jako 3

    # Tworzenie dodatkowych etykiet dla dopasowań
    additional_labels_train.extend([labels[matches.index((user_seq, db_seq))]] * len(user_seq.split(" i ")))

def get_audio_files_in_folder(folder_path):
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.ons', '.onset')):
                audio_files.append(os.path.join(root, file))
    return audio_files

# Ścieżki do plików .wav
database_folder = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\note_detection\FTIMSET\wav\Leveau"

# Ścieżki do plików .ons
database_onsets = r"C:\Users\gosia\OneDrive - vus.hr\Desktop\note_detection\FTIMSET\ons\Leveau"

# Pliki .wav
database = [{'file_path': file} for file in get_audio_files_in_folder(database_folder)]

# Pliki .ons
database_onsets_files = [{'file_path': file} for file in get_audio_files_in_folder(database_onsets)]

labels_edit_distance, labels_alignment_to_line = compare_with_database(user_folder_path, database, database_onsets_files)

# Definicja modelu sieci neuronowej
model = Sequential([
    Dense(64, activation='relu', input_shape=(len(user_sequences[0]),)),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')  # Wyjście: odległość edycyjna i ocena dopasowania do prostej
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='mean_squared_error')

# Trenowanie modelu
model.fit(user_sequences_scaled, np.column_stack((labels_edit_distance, labels_alignment_to_line)), epochs=100, batch_size=32, validation_split=0.2)

# Funkcja do oceny dopasowania zapytania użytkownika
def evaluate_query(query_sequence, db_sequences):
    # Normalizacja sekwencji użytkownika
    query_sequence_scaled = scaler.transform(np.array([query_sequence]))

    # Prognozowanie odległości edycyjnej i oceny dopasowania do prostej dla wszystkich sekwencji w bazie danych
    predictions = model.predict(query_sequence_scaled)

    # Znalezienie najlepiej dopasowanej sekwencji na podstawie prognoz
    best_match_index = np.argmin(predictions[:, 0])  # Minimalna odległość edycyjna
    best_match_alignment = np.argmax(predictions[:, 1])  # Maksymalna ocena dopasowania do prostej

    # Zwrócenie najlepiej dopasowanej sekwencji i jej oceny dopasowania
    return db_sequences[best_match_index], best_match_alignment

# Pętla interakcyjna do wprowadzania zapytań użytkownika
while True:
    user_input = input("Podaj ścieżkę do pliku dźwiękowego użytkownika (wprowadź 'q' aby zakończyć): ")
    if user_input.lower() == 'q':
        break
    try:
        # Wczytanie sekwencji dźwiękowej z pliku
        y, sr = librosa.load(user_input)
        query_sequence = calculate_onset_strength(y, sr)
        best_match, alignment_score = evaluate_query(query_sequence, db_sequences)
        print(f"Najlepiej dopasowana sekwencja dla zapytania {user_input}: {best_match}, ocena dopasowania: {alignment_score}")
    except Exception as e:
        print(f"Błąd: {e}")
