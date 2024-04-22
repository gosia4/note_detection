import numpy as np
import matplotlib.pyplot as plt
import librosa
from fastdtw import fastdtw
from edit_distance import evaluate_alignment_to_line


# my dtw function only using numpy
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)

    # Inicjalizacja macierzy kosztów
    cost_matrix = np.zeros((n, m))

    # Obliczenia pierwszego wiersza i pierwszej kolumny
    cost_matrix[0, 0] = np.abs(s1[0] - s2[0])

    for i in range(1, n):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + np.abs(s1[i] - s2[0])  # Wypełnianie pierwszego wiersza czyli usunięcie w sekwencji s2.

    for j in range(1, m):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + np.abs(s1[0] - s2[j])  # Wypełnianie pierwszej kolumny czyli usunięcie w sekwencji s1.

    # Wypełnianie reszty macierzy czyli suma kosztu aktualnego dopasowania i minimum (czyli minimalizując koszt)
    # spośród trzech możliwych operacji:
    # usunięcia w sekwencji s1, usunięcia w sekwencji s2 lub zastąpienia w sekwencji s1 i s2.
    for i in range(1, n):
        for j in range(1, m):
            cost_matrix[i, j] = np.abs(s1[i] - s2[j]) + min(
                cost_matrix[i - 1, j],       # operacja usunięcia w sekwencji s1, s1 dłuższe niż s2
                cost_matrix[i, j - 1],       # operacja usunięcia w sekwencji s2
                cost_matrix[i - 1, j - 1]    # operacja zastąpienia w sekwencji s1 i s2, różnica w wartościach w s1 i s2
            )

    # Obliczanie odległości DTW, ostatni element zawiera łączny koszt dopasowania
    dtw_distance = cost_matrix[n - 1, m - 1]

    return dtw_distance



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


# dtw for onset strength
def calculate_dtw_librosa(user_sequence, db_sequence, file_name, show=False, save=False, normalize=True):
    # normalize the amplitude, so that if the user sequence has different amplitude, it will match better dtw
    # if normalize:
    #     user = librosa.util.normalize(user)
    #     db = librosa.util.normalize(db)

    distance, path = librosa.sequence.dtw(user_sequence, db_sequence)
    print("Path shape:", path.shape)

    if show:
        # from librosa
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(distance, x_axis='frames', y_axis='frames', ax=ax[0])
        ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
        ax[0].plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
        ax[0].legend()
        fig.colorbar(img, ax=ax[0])
        ax[1].plot(distance[-1, :] / path.shape[0])
        ax[1].set(xlim=[0, db_sequence.shape[0]], ylim=[0, 2], title='Matching cost function: ' + file_name)
        if save:
            plt.savefig('dtw' + file_name + '.png')
        plt.show()

    # mean_distance = evaluate_alignment_to_line(path.T, user_sequence, db_sequence)
    # print(f"Średnia odległość punktów na ścieżce od prostej: {mean_distance}")
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
    # if len(user) == 0 or len(db) == 0:
    #     raise ValueError("Empty sequence(s) provided")
    # user_sequence = np.zeros(len(user))
    # db_sequence = np.zeros(len(user))  # length of the db files the same as the user files

    max_user_index = int(max(user)) if len(user) > 0 else 0
    max_db_index = len(db)

    user_sequence = np.zeros(max_user_index + 1)
    db_sequence = np.zeros(max_db_index)
    # Add random disturbance to the user sequence, so that it will imitate the mistakes from the user
    for i in range(len(user_sequence)):
        # Generate a random number between 0 and 1
        random_value = np.random.rand()
        # Choose a threshold, for example 0.8????
        threshold = 0.8
        # If the random value is greater than the threshold, set the onset
        if random_value > threshold:
            user_sequence[i] = 1

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


# dopasowanie długości sekwencji użytkownika do długości w bazie danych
def calculate_dtw_librosa_onsets_próbkowanie(user, db, file_name, show=False, save=False):
    max_db_index = len(db)
    # tworzenie sekwencji o długości max, która jest a bazie danych
    user_sequence = np.zeros(max_db_index)
    db_sequence = np.zeros(max_db_index)

    # Normalize user sequence length to match the length of the database sequence
    # próbki, które odpowiadają uderzeniom w sekwencji użytkownika, mają wartość 1
    if len(user) > 0:
        max_user_index = int(max(user))
        user_sequence[:max_user_index] = 1

    # Add random disturbance to the user sequence
    for i in range(len(user_sequence)):
        random_value = np.random.rand()
        threshold = 0.8
        if random_value > threshold:
            user_sequence[i] = 1  # tu wystąpiło uderzenie

    # każde uderzenie jest reprezentowane przez próbkę w db_sequence,
    # która odpowiada indeksowi uderzenia, i tu mamy binarną reprezentację
    for db_onset in db:
        if int(db_onset) < max_db_index:  # Checking if the index is in range
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
        ax[1].set(xlim=[0, max_db_index], ylim=[0, 2], title='Matching cost function: ' + file_name)
        if save:
            plt.savefig('dtw' + file_name + '.png')
        plt.show()

    average_distance = np.mean(distance)

    return average_distance, path