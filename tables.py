# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Tworzymy przykładowe dane
# data = {
#     'person': ['person2', 'person 4', 'person 8'],
#     'piece 12': ['12, 15, 22, 7, 20', '12, 7, 15, 22, 16', '12, 15, 22, 7, 20'],
#     'piece 16': ['12, 15, 22, 7, 20', '12, 22, 7, 15, 16', '12, 15, 22, 7, 20'],
#     'piece 20': ['12, 15, 22, 7, 20', '12, 15, 22, 7, 20', '15, 22, 12, 20, 7'],
#     'piece 22': ['15, 22, 12, 20, 7', '12, 15, 22, 7, 20', '12, 7, 22, 15, 20'],
#     'piece 30': ['30, 15, 20, 12, 22', '15, 30, 20, 22, 12', '15,12,22,20,30'],
#     'piece 44': ['15, 30, 22, 20, 12', '15, 12, 22, 20, 30', '15, 12, 22, 20, 7']
# }
#
# # Konwersja danych do DataFrame
# df = pd.DataFrame(data)
#
# # Funkcja do sprawdzania, czy właściwy utwór jest w top5
# def check_in_top5(row, piece):
#     top5 = list(map(int, row.split(',')))  # Konwersja do listy liczb
#     piece_num = int(piece.split(' ')[1])  # Wyciągnięcie numeru utworu
#     return 1 if piece_num in top5 else 0
#
# # Iterujemy po każdej kolumnie z utworami i przekształcamy dane
# for column in df.columns[1:]:
#     df[column] = df[column].apply(lambda x: check_in_top5(x, column))
#
# # Tworzenie wykresu heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(df.set_index('person'), annot=True, cmap='YlGnBu', cbar=True)
#
# # Dodanie tytułu i etykiet
# plt.title('Is appropriate piece matching th query?')
# plt.xlabel('Pieces')
# plt.ylabel('Queries')
#
# # Pokaż wykres
# plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Funkcja do przekształcenia danych tekstowych w liczby i wygenerowania wykresu
def generate_heatmap(data, title):
    df = pd.DataFrame(data)

    # Funkcja do sprawdzania, czy dany utwór jest w top5
    def check_in_top5(row, piece):
        top5 = list(map(int, row.split(',')))  # Konwersja do listy liczb
        piece_num = int(piece.split(' ')[1])  # Wyciągnięcie numeru utworu
        return 1 if piece_num in top5 else 0

    # Iterujemy po każdej kolumnie z utworami i przekształcamy dane
    for column in df.columns[1:]:
        df[column] = df[column].apply(lambda x: check_in_top5(x, column))

    # Tworzenie wykresu heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.set_index('person'), annot=True, cmap='YlGnBu', cbar=True)

    # Dodanie tytułu i etykiet
    plt.title(title)
    plt.xlabel('Pieces')
    plt.ylabel('Queries')

    # Pokaż wykres
    plt.show()

# Dane dla funkcji 1
data_fun1 = {
    'person': ['person1', 'person 2', 'person 3'],
    'piece 12': ['30, 15, 20,12,  22', '30, 15, 20,12,  22', '30, 15, 20,12,  22'],
    'piece 16': ['30, 15, 20,12,  22', '30, 15, 20,12,  22', '30, 15, 20,12,  22'],
    'piece 20': ['30, 15, 20,12,  22', '30, 15, 20,12,  22', '30, 15, 20,12,  22'],
    'piece 22': ['30, 15, 20,12,  22', '30, 15, 20,12,  22', '30, 15, 20,12,  22'],
    'piece 30': ['30, 15, 20,12,  22', '30, 15, 20,12,  22', '30, 15, 20,12,  22'],
    'piece 44': ['30, 15, 20,12,  22', '30, 15, 20,12,  22', '30, 15, 20,12,  22']
}

data_fun2 = {
    'person': ['person1', 'person 2', 'person 3'],
    'piece 12': ['12, 15, 22, 7, 20', '12, 7, 15, 22, 16', '12, 15, 22, 7, 20'],
    'piece 16': ['12, 15, 22, 7, 20', '12, 22, 7, 15, 16', '12, 15, 22, 7, 20'],
    'piece 20': ['12, 15, 22, 7, 20', '12, 15, 22, 7, 20', '15, 22, 12, 20, 7'],
    'piece 22': ['15, 22, 12, 20, 7', '12, 15, 22, 7, 20', '12, 7, 22, 15, 20'],
    'piece 30': ['30, 15, 20, 12, 22', '15, 30, 20, 22, 12', '15,12,22,20,30'],
    'piece 44': ['15, 30, 22, 20, 12', '15, 12, 22, 20, 30', '15, 12, 22, 20, 7']
}

# Dane dla funkcji 3
data_fun3 = {
    'person': ['person1', 'person 2', 'person 3'],
    'piece 12': ['26, 38, 46, 23, 21', '46, 26, 32, 38, 23', '26, 32, 28, 23, 46'],
    'piece 16': ['40, 26, 46, 32, 23', '27, 26, 23, 32, 46', '26, 21, 46, 23, 38'],
    'piece 20': ['46, 32, 38, 26, 23', '38, 32, 26, 23, 46', '46, 26, 32, 38, 23'],
    'piece 22': ['29, 26, 46, 23, 38', '10, 26, 23, 38, 46', '32, 26, 23, 46, 37'],
    'piece 30': ['43, 44, 46, 38, 32', '13, 19, 25, 26, 46', '26, 46, 23, 32, 38'],
    'piece 44': ['46, 26, 23, 38, 14', '46, 26, 38, 23, 13', '26, 23, 38, 46, 32']
}

# Dane dla funkcji 5
data_fun5 = {
    'person': ['person1', 'person 2', 'person 3'],
    'piece 12': ['12, 36, 16, 3, 30', '15, 30, 9, 20, 12', '36, 30, 3, 20, 15'],
    'piece 16': ['30, 16, 20, 15, 7', '15, 16, 30, 7, 22', '30, 15, 41, 36, 12'],
    'piece 20': ['30, 15, 22, 41, 36', '30, 16, 7, 3, 12', '15, 30, 41, 12, 16'],
    'piece 22': ['30, 15, 41, 20, 4', '36, 16, 22, 41, 3', '30, 7, 22, 43, 15'],
    'piece 30': ['30, 12, 11, 36, 15', '30, 36, 20, 12, 22', '30, 20, 12, 15, 7'],
    'piece 44': ['12, 30, 3, 16, 15', '30, 15, 12, 22, 3', '36, 22, 30, 7, 16']
}

# Dane dla funkcji 6
data_fun6 = {
    'person': ['person1', 'person 2', 'person 3'],
    'piece 12': ['30, 12, 16, 7, 15', '30, 12, 16, 7, 15', '30, 12, 16, 7, 15'],
    'piece 16': ['30, 12, 16, 7, 15', '30, 12, 16, 7, 15', '30, 12, 16, 7, 15'],
    'piece 20': ['30, 12, 16, 7, 15', '30, 12, 16, 7, 15', '30, 12, 16, 7, 15'],
    'piece 22': ['30, 12, 7, 16, 15', '30, 12, 16, 7, 15', '30, 12, 16, 7, 15'],
    'piece 30': ['30, 12, 7, 16, 15', '30, 12, 16, 7, 15', '30, 12, 16, 7, 15'],
    'piece 44': ['30, 12, 16, 7, 15', '30, 12, 16, 7, 15', '30, 12, 16, 7, 15']
}

# Dane dla funkcji 7
data_fun7 = {
    'person': ['person1', 'person 2', 'person 3'],
    'piece 12': ['40, 15, 20, 48, 1', '20, 22, 14, 19, 16', '10, 41, 48, 8, 4'],
    'piece 16': ['16, 7, 42, 3, 10', '40, 14, 16, 19, 42', '40, 36, 12, 48, 3'],
    'piece 20': ['46, 16, 47, 14, 48', '10, 45, 34, 43, 5', '38, 42, 12, 48, 7'],
    'piece 22': ['34, 40, 19, 12, 2', '22, 10, 14, 16, 46', '42, 29, 47, 4, 15'],
    'piece 30': ['5, 31, 3, 11, 8', '41, 16, 19, 10, 30', '47, 44, 39, 3, 26'],
    'piece 44': ['14, 12, 35, 42, 18', '14, 36, 10, 45, 13', '18, 5, 30, 7, 46']
}

# Dane dla funkcji 8
data_fun8 = {
    'person': ['person1', 'person 2', 'person 3'],
    'piece 12': ['12, 16, 7, 30, 15', '12, 16, 7, 30, 15', '12, 16, 7, 15, 30'],
    'piece 16': ['12, 16, 7, 15, 30', '12, 16, 7, 30, 15', '12, 16, 7, 30, 15'],
    'piece 20': ['12, 16, 7, 15, 30', '12, 16, 7, 30, 15', '12, 16, 7, 15, 30'],
    'piece 22': ['12, 16, 7, 30, 15', '12, 16, 7, 30, 15', '12, 16, 7, 30, 15'],
    'piece 30': ['12, 16, 7, 30, 15', '12, 16, 7, 30, 15', '12, 16, 7, 30, 15'],
    'piece 44': ['12, 16, 7, 30, 15', '12, 16, 7, 30, 15', '12, 16, 7, 30, 15']
}

# Generowanie wykresów dla każdej funkcji
generate_heatmap(data_fun1, "Function 1")
generate_heatmap(data_fun2, "Function 2")
generate_heatmap(data_fun3, "Function 3")
generate_heatmap(data_fun5, "Function 5")
generate_heatmap(data_fun6, "Function 6")
generate_heatmap(data_fun7, "Function 7")
generate_heatmap(data_fun8, "Function 8")
