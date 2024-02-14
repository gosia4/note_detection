import numpy as np

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
