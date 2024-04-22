import librosa
import matplotlib.pyplot as plt


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
