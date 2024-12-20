import librosa
import matplotlib.pyplot as plt
import numpy as np


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


def detect_onsets_dynamic_threshold(onset_strength, threshold_factor=0.02, fs=44100):
    onsets = []  # List with times of the onsets

    # Average and standard deviation of the onsets
    average_strength = np.mean(onset_strength)
    std_strength = np.std(onset_strength)

    # Dynamic treshlod - arythmetic mean and standard deviation
    threshold = average_strength + threshold_factor * std_strength

    # Onset detection
    for i in range(1, len(onset_strength)):
        if onset_strength[i] > threshold >= onset_strength[i - 1]:
            onsets.append(librosa.frames_to_time(i, sr=fs))

    return onsets

# for .ons files
def load_onsets(file_path):
    with open(file_path, 'r') as file:
        onsets = [float(line.strip()) for line in file]
    return onsets
