import numpy as np
import matplotlib.pyplot as plt
import librosa


def calculate_dtw_librosa_onsets_edit_distance(user, db, file_name, show=False, save=False):
    user_sequence = np.zeros(len(user))
    db_sequence = np.zeros(len(db))  # length of db, or should length of the db files be the same as the user files???

    # db_sequence = np.zeros(max(len(user), len(db)))
    # Add random disturbance to the user sequence, so that when the user claps not precisely
    for i in range(len(user_sequence)):
        # random number between 0 and 0.2
        random_value = np.random.uniform(0, 0.2)

        # Choose a threshold, for example 0.8
        threshold = 0.8
        # If the random value is greater than the threshold, set the onset
        if random_value > threshold:
            user_sequence[i] = 1

    # Jeśli uderzenie w sekwencji użytkownika, próbka w user_sequence jest 1
    # Jeśli uderzenie w bazie danych, próbka w db_sequence jest ustawiana na 1.
    for user_onset in user:
        if 0 <= int(user_onset) < len(user_sequence):
            user_sequence[int(user_onset)] = 1
    for db_onset in db:
        if 0 <= int(db_onset) <= len(db_sequence):  # Checking if the index is in range
            db_sequence[int(db_onset)] = 1

    # Compute DTW distance matrix using custom distance measure (edit distance)
    distance = np.zeros((len(user_sequence), len(db_sequence)))
    for i in range(len(user_sequence)):
        for j in range(len(db_sequence)):
            distance[i, j] = abs(user_sequence[i] - db_sequence[
                j])  # Edit distance measure (wartość bezwzględna z odległości między sekwencjami

    # Calculate cumulative cost matrix
    for i in range(1, len(user_sequence)):
        distance[i, 0] += distance[
            i - 1, 0]  # Dla każdego wiersza, wartość w kolumnie zerowej (distance[i, 0]) jest zwiększana o wartość z poprzedniego wiersza w tej samej kolumnie (distance[i - 1, 0]).
        # Oznacza to, że każda komórka w kolumnie zerowej będzie sumą wszystkich komórek powyżej niej.
    for j in range(1, len(db_sequence)):
        distance[0, j] += distance[0, j - 1]
    for i in range(1, len(user_sequence)):
        for j in range(1, len(db_sequence)):
            distance[i, j] += np.min([distance[i - 1, j], distance[i, j - 1], distance[i - 1, j - 1]])
    #  Dla każdej komórki (distance[i, j]), dodawana jest minimalna wartość z trzech sąsiadujących komórek:
    #  komórki powyżej (distance[i - 1, j]), komórki po lewej (distance[i, j - 1]) i komórki z górnego lewego rogu (distance[i - 1, j - 1]).
    #  To jest optymalny koszt dopasowania do danej komórki.

    # Compute optimal path through backtracking
    path = []
    i, j = len(user_sequence) - 1, len(db_sequence) - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if distance[i - 1, j - 1] <= min(distance[i - 1, j], distance[i, j - 1]):
                i, j = i - 1, j - 1
            elif distance[i, j - 1] <= min(distance[i - 1, j], distance[i - 1, j - 1]):
                j -= 1
            else:
                i -= 1

    path.append((0, 0))

    # Reverse the path
    path.reverse()

    if show:
        plt.imshow(distance, origin='lower', cmap='gray_r', aspect='auto', interpolation='nearest')
        plt.plot([x[1] for x in path], [x[0] for x in path], color='yellow')
        plt.title('DTW cost and optimal path')
        plt.xlabel(file_name)
        plt.ylabel('User sequence')
        plt.colorbar()
        if save:
            plt.savefig('dtw' + file_name + '.png')
        plt.show()

    average_distance = distance[-1, -1] / len(path)

    return average_distance, path


def calculate_edit_distance(user_sequence, db_sequence, Tm, file_name, show=False, save=False):
    # Inicjalizacja macierzy odległości, +1 dla przypadku gdy byłby pusty sequence
    distance = np.zeros((len(user_sequence) + 1, len(db_sequence) + 1))
    path = np.zeros((len(user_sequence) + 1, len(db_sequence) + 1, 2), dtype=int)

    # Obliczenie kosztów wstawiania i usuwania:
    # Koszty wstawiania i usuwania - odległość od pustej sekwencji, czyli wstawienie lub usunięcie wszystkich elementów.
    for i in range(len(user_sequence) + 1):
        distance[i, 0] = i
    for j in range(len(db_sequence) + 1):
        distance[0, j] = j

    for i in range(len(user_sequence)):
        noise = np.random.uniform(0, 0.2)
        user_sequence[i] += noise  # Dodawanie szumu do aktualnej wartości sekwencji użytkownika

    # Obliczanie odległości edycyjnej
    for i in range(1, len(user_sequence) + 1):
        for j in range(1, len(db_sequence) + 1):
            if user_sequence[i - 1] == db_sequence[j - 1]:
                cost = 0  # Dopasowanie - identyczne to jest 0
            else:
                if db_sequence[j - 1] != 0:
                    # Obliczanie kosztu zastąpienia - proporcja IOI z artykułu
                    ratio = user_sequence[i - 1] / db_sequence[j - 1]
                    if 1 < ratio < Tm or 1 < 1 / ratio < Tm:
                        cost = 1 - min(ratio, 1 / ratio)  # wybieramy wartość mniejszą, albo ratio, albo 1/ratio
                    else:
                        cost = 1  # Koszt zastąpienia
                else:
                    cost = 1  # Koszt zastąpienia, gdy db_sequence[j - 1] == 0

            # Obliczenie kosztu edycji dla danej komórki
            min_cost = np.min([distance[i - 1, j] + 1,  # Usunięcie (pomijanie elementu sekwencji)
                               distance[i, j - 1] + 1,  # Wstawienie - dodanie nowego elementu do jednej sekwencji
                               distance[i - 1, j - 1] + cost])  # Zastąpienie -zamianę jednego elementu na inny
            distance[i, j] = min_cost

            # path
            if min_cost == distance[i - 1, j] + 1:
                path[i, j] = [i - 1, j]
            elif min_cost == distance[i, j - 1] + 1:
                path[i, j] = [i, j - 1]
            else:
                path[i, j] = [i - 1, j - 1]

                #TODO z librosy path:

    distance_librosa, path = librosa.sequence.dtw(user_sequence, db_sequence)

    if show:
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(distance_librosa, x_axis='frames', y_axis='frames', ax=ax[0])
        ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
        ax[0].plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
        ax[0].legend()
        fig.colorbar(img, ax=ax[0])
        ax[1].plot(distance_librosa[-1, :] / path.shape[0])
        # ax[1].set(xlim=[0, db.shape[0]], ylim=[0, 2], title='Matching cost function: ' + file_name)
        ax[1].set(xlim=[0, len(db_sequence)], ylim=[0, 2], title='Matching cost function: ' + file_name)
        if save:
            plt.savefig('dtw' + file_name + '.png')
        plt.show()

    # if show:
    #     fig, ax = plt.subplots(nrows=2, sharex=True)
    #     img = librosa.display.specshow(distance, x_axis='frames', y_axis='frames', ax=ax[0])
    #     ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
    #     ax[0].plot(path[:, 1], path[:, 0], label='Optimal path', color='y')
    #     ax[0].legend()
    #     fig.colorbar(img, ax=ax[0])
    #     ax[1].plot(distance[-1, :] / path.shape[0])
    #     # ax[1].set(xlim=[0, db.shape[0]], ylim=[0, 2], title='Matching cost function: ' + file_name)
    #     ax[1].set(xlim=[0, len(db_sequence)], ylim=[0, 2], title='Matching cost function: ' + file_name)
    #     if save:
    #         plt.savefig('dtw' + file_name + '.png')
    #     plt.show()


    # Zwrócenie odległości edycyjnej (ostatnia komórka macierzy) i ścieżki (None bo nie jest wykorzystywana)
    return distance[len(user_sequence), len(db_sequence)], path


#calculate_edit_distance z rysowaniem macierzy kosztów i dopasowywaniem sekwecji do ścieżki, aby była jak najbardziej zbliżona do linii prostej
def calculate_edit_distance2(user_sequence, db_sequence, Tm, file_name, show=False, save=False):
    # Inicjalizacja macierzy odległości, +1 dla przypadku gdy byłby pusty sequence
    distance = np.zeros((len(user_sequence) + 1, len(db_sequence) + 1))
    path = np.zeros((len(user_sequence) + 1, len(db_sequence) + 1, 2), dtype=int)

    # Obliczenie kosztów wstawiania i usuwania:
    # Koszty wstawiania i usuwania - odległość od pustej sekwencji, czyli wstawienie lub usunięcie wszystkich elementów.
    for i in range(len(user_sequence) + 1):
        distance[i, 0] = i
    for j in range(len(db_sequence) + 1):
        distance[0, j] = j

    for i in range(len(user_sequence)):
        noise = np.random.uniform(0, 0.2)
        user_sequence[i] += noise  # Dodawanie szumu do aktualnej wartości sekwencji użytkownika

    # Obliczanie odległości edycyjnej
    # for i in range(1, len(user_sequence) + 1):
    #     for j in range(1, len(db_sequence) + 1):
    #         if user_sequence[i - 1] == db_sequence[j - 1]:
    #             cost = 0  # Dopasowanie - identyczne to jest 0
    #         else:
    #             if db_sequence[j - 1] != 0:
    #                 # Obliczanie kosztu zastąpienia - proporcja IOI z artykułu
    #                 ratio = user_sequence[i - 1] / db_sequence[j - 1]
    #                 if 1 < ratio < Tm or 1 < 1 / ratio < Tm:
    #                     cost = 1 - min(ratio, 1 / ratio)  # wybieramy wartość mniejszą, albo ratio, albo 1/ratio
    #                 else:
    #                     cost = 1  # Koszt zastąpienia
    #             else:
    #                 cost = 1  # Koszt zastąpienia, gdy db_sequence[j - 1] == 0
    # Obliczanie odległości edycyjnej
    for i in range(1, len(user_sequence) + 1):
        for j in range(1, len(db_sequence) + 1):
            if user_sequence[i - 1] == db_sequence[j - 1]:
                cost = 0.5  # Koszt dopasowania
            else:
                if db_sequence[j - 1] != 0:
                    # Obliczanie kosztu zastąpienia - proporcja IOI z artykułu
                    ratio = user_sequence[i - 1] / db_sequence[j - 1]
                    if 1 < ratio < Tm or 1 < 1 / ratio < Tm:
                        cost = 0.5 - min(ratio, 1 / ratio)  # Koszt niezgodności
                    else:
                        cost = -0.5  # Koszt zastąpienia
                else:
                    cost = -0.5  # Koszt zastąpienia, gdy db_sequence[j - 1] == 0

            # Obliczenie kosztu edycji dla danej komórki
            # min_cost = np.min([distance[i - 1, j] + 1,  # Usunięcie (pomijanie elementu sekwencji)
            #                    distance[i, j - 1] + 1,  # Wstawienie - dodanie nowego elementu do jednej sekwencji
            #                    distance[i - 1, j - 1] + cost])  # Zastąpienie -zamianę jednego elementu na inny
            min_cost = np.min([distance[i - 1, j] - 0.5,  # Usunięcie (pomijanie elementu sekwencji)
                               distance[i, j - 1] - 0.5,  # Wstawienie - dodanie nowego elementu do jednej sekwencji
                               distance[i - 1, j - 1] + cost])

            distance[i, j] = min_cost


            # path
            if min_cost == distance[i - 1, j] + 1:
                path[i, j] = [i - 1, j]
                # print(f"Appending to path: {[i - 1, j]}")  # Add this line
            elif min_cost == distance[i, j - 1] + 1:
                path[i, j] = [i, j - 1]
            else:
                path[i, j] = [i - 1, j - 1]
    print(f"user_sequence length: {len(user_sequence)}")
    print(f"db_sequence length: {len(db_sequence)}")
    print(f"path shape: {path.shape}")
    # if show:
    #     # Rysowanie macierzy kosztów
    #     fig, ax = plt.subplots()
    #     ax.imshow(distance, cmap='viridis', interpolation='nearest')
    #     ax.set_title('Cost Matrix')
    #     ax.set_xlabel('User Sequence')
    #     ax.set_ylabel('Database Sequence: ' + file_name)
    #     if save:
    #         plt.savefig('dtw' + file_name + '.png')
    #     plt.show()
    # plot_alignment(user_sequence, db_sequence, path)

    # Ocenianie dopasowania do prostej
    mean_distance = evaluate_alignment_to_line(path[:, :, :2], user_sequence, db_sequence)
    print(f"Średnia odległość punktów na ścieżce od prostej: {mean_distance}")

    # Zwrócenie odległości edycyjnej, ścieżki i oceny dopasowania do prostej
    return distance[len(user_sequence), len(db_sequence)], path, mean_distance


def evaluate_alignment_to_line(path, user_sequence, db_sequence):
    """
    Ocenia dopasowanie sekwencji użytkownika do sekwencji z bazy danych na podstawie ścieżki najlepszego dopasowania.
    Funkcja oblicza średnią odległość punktów na ścieżce od prostej łączącej początek i koniec sekwencji.

    Args:
        path (np.ndarray): Tablica zawierająca ścieżkę najlepszego dopasowania (kolumny: x, y).
        user_sequence (np.ndarray): Sekwencja użytkownika.
        db_sequence (np.ndarray): Sekwencja z bazy danych.

    Returns:
        float: Średnia odległość punktów na ścieżce od prostej.
    """

    # Wyznaczenie współczynników prostej
    m = (db_sequence[-1] - db_sequence[0]) / (user_sequence[-1] - user_sequence[0])
    b = db_sequence[0] - m * user_sequence[0]

    # Obliczenie odległości punktów na ścieżce od prostej
    # distances = []
    # for i in range(len(path)):
    #     for j in range(len(path[i])):
    #         x, y = path[i, j]
    #         # for x, y in path:
    #         x_user = user_sequence[x]
    #         y_user = db_sequence[y]
    #         distance = np.abs(y_user - (m * x_user + b))
    #         distances.append(distance)
    distances = []
    for i in range(len(path)):
        for j in range(len(path[i])):
            if i < len(user_sequence) and j < len(db_sequence):  # Check for valid indices
                x, y = path[i, j]
                # point = path[i]
                # x = point[0]
                # y = point[1]
                # if x < len(user_sequence) and y < len(db_sequence):
                #     x_user = user_sequence[x]
                #     y_user = db_sequence[y]
                #     # reszta kodu
                # else:
                #     pass
                x_user = user_sequence[x]
                y_user = db_sequence[y]
                distance = np.abs(y_user - (m * x_user + b))
                distances.append(distance)
            else:
                # Handle gaps (optional: assign a penalty value)
                pass
    # Obliczenie średniej odległości
    mean_distance = np.mean(distances)

    return mean_distance

def plot_alignment(user_sequence, db_sequence, path):
    """
    Rysuje wykres najlepszego dopasowania sekwencji użytkownika i sekwencji z bazy danych.

    Args:
        user_sequence (np.ndarray): Sekwencja użytkownika.
        db_sequence (np.ndarray): Sekwencja z bazy danych.
        path (np.ndarray): Macierz ścieżki dopasowania.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Rysowanie sekwencji użytkownika
    ax.plot(user_sequence, label='Sekwencja użytkownika', color='blue')

    # Rysowanie sekwencji z bazy danych
    ax.plot(db_sequence, label='Sekwencja z bazy danych', color='green')

    # Rysowanie ścieżki dopasowania
    x_path = [x[0] for x in path[:, 1:]]
    y_path = [x[1] for x in path[:, 1:]]
    ax.plot(x_path, y_path, color='red', linewidth=2, alpha=0.7)

    # Dodanie legendy i tytułu
    ax.legend()
    ax.set_title('Najlepsze dopasowanie sekwencji')

    # Wyświetlenie wykresu
    plt.show()


def calculate_edit_distance4(user_sequence, db_sequence, Tm, file_name, show=False, save=False):
    # Initialize distance matrix, +1 for empty sequence
    distance = np.zeros((len(user_sequence) + 1, len(db_sequence) + 1))
    path = np.zeros((len(user_sequence) + 1, len(db_sequence) + 1, 2), dtype=int)

    # Calculate insertion and deletion costs
    insertion_cost = -1  # Fixed cost for inserting an element
    deletion_cost = -1  # Fixed cost for deleting an element
    for i in range(len(user_sequence) + 1):
        distance[i, 0] = i * insertion_cost
    for j in range(len(db_sequence) + 1):
        distance[0, j] = j * deletion_cost

    for i in range(len(user_sequence)):
        noise = np.random.uniform(0, 0.2)
        user_sequence[i] += noise  # Add noise to current user sequence value

    # Calculate edit distance
    for i in range(1, len(user_sequence) + 1):
        for j in range(1, len(db_sequence) + 1):
            if user_sequence[i - 1] == db_sequence[j - 1]:
                cost = 0  # Match - identical is 0
            else:
                if db_sequence[j - 1] != 0:
                    # Calculate substitution cost - IOI ratio from paper
                    ratio = user_sequence[i - 1] / db_sequence[j - 1]
                    if 1 < ratio < Tm or 1 < 1 / ratio < Tm:
                        cost = 1 - min(ratio, 1 / ratio)  # Choose smaller value, either ratio or 1/ratio
                    else:
                        cost = 1  # Substitution cost
                else:
                    cost = 1  # Substitution cost when db_sequence[j - 1] == 0

            # Calculate edit cost for current cell
            min_cost = np.min([distance[i - 1, j] + deletion_cost,  # Deletion (skip sequence element)
                               distance[i, j - 1] + insertion_cost,  # Insertion - add new element to one sequence
                               distance[i - 1, j - 1] + cost])  # Substitution - replace one element with another
            distance[i, j] = min_cost

            # Path
            if min_cost == distance[i - 1, j] + deletion_cost:
                path[i, j] = [i - 1, j]
            elif min_cost == distance[i, j - 1] + insertion_cost:
                path[i, j] = [i, j - 1]
            else:
                path[i, j] = [i - 1, j - 1]

    print(f"user_sequence length: {len(user_sequence)}")
    print(f"db_sequence length: {len(db_sequence)}")
    print(f"path shape: {path.shape}")

    # Evaluate alignment to line
    mean_distance = evaluate_alignment_to_line(path[:, :, :2], user_sequence, db_sequence)
    print(f"Średnia odległość punktów na ścieżce od prostej: {mean_distance}")

    # Return edit distance, path, and alignment to line score
    return distance[len(user_sequence), len(db_sequence)], path, mean_distance

def calculate_edit_distance3(user_sequence, db_sequence, Tm, file_name, show=False, save=False):
    # Initialize distance matrix, +1 for empty sequence
    distance = np.zeros((len(user_sequence) + 1, len(db_sequence) + 1))
    path = np.zeros((len(user_sequence) + 1, len(db_sequence) + 1, 2), dtype=int)

    # Calculate insertion and deletion costs
    for i in range(len(user_sequence) + 1):
        distance[i, 0] = i
    for j in range(len(db_sequence) + 1):
        distance[0, j] = j

    for i in range(len(user_sequence)):
        noise = np.random.uniform(0, 0.2)
        user_sequence[i] += noise  # Add noise to current user sequence value

    # Calculate edit distance
    for i in range(1, len(user_sequence) + 1):
        for j in range(1, len(db_sequence) + 1):
            if user_sequence[i - 1] == db_sequence[j - 1]:
                cost = 0  # Match - identical is 0
            else:
                if db_sequence[j - 1] != 0:
                    # Calculate substitution cost - IOI ratio from paper
                    ratio = user_sequence[i - 1] / db_sequence[j - 1]
                    if 1 < ratio < Tm or 1 < 1 / ratio < Tm:
                        cost = 1 - min(ratio, 1 / ratio)  # Choose smaller value, either ratio or 1/ratio
                    else:
                        cost = 1  # Substitution cost
                else:
                    cost = 1  # Substitution cost when db_sequence[j - 1] == 0

            # Calculate edit cost for current cell
            min_cost = np.min([distance[i - 1, j] + 1,  # Deletion (skip sequence element)
                               distance[i, j - 1] + 1,  # Insertion - add new element to one sequence
                               distance[i - 1, j - 1] + cost])  # Substitution - replace one element with another
            distance[i, j] = min_cost

            # Path
            if min_cost == distance[i - 1, j] + 1:
                path[i, j] = [i - 1, j]
            elif min_cost == distance[i, j - 1] + 1:
                path[i, j] = [i, j - 1]
            else:
                path[i, j] = [i - 1, j - 1]

    print(f"user_sequence length: {len(user_sequence)}")
    print(f"db_sequence length: {len(db_sequence)}")
    print(f"path shape: {path.shape}")

    # Evaluate alignment to line
    mean_distance = evaluate_alignment_to_line(path[:, :, :2], user_sequence, db_sequence)
    print(f"Średnia odległość punktów na ścieżce od prostej: {mean_distance}")

    # Return edit distance, path, and alignment to line score
    return distance[len(user_sequence), len(db_sequence)], path, mean_distance
