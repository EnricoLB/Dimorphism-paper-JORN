import os
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from tqdm import tqdm

# ======================
# PARAMETERS
# ======================
SAMPLE_RATE = 44100
N_MFCC = 12
N_FFT = 512
FMAX = 5000
BUFFER_SEC = 0.3

# ======================
# FILTER FUNCTION
# ======================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# ======================
# LOAD DATA
# ======================
metadata = pd.read_csv("data/metadata/annotations.csv")
audio_dir = "data/audio/"

# Fix filenames
metadata["Filename"] = metadata["Begin_File"].str.replace("mpeg|mp3", "wav", regex=True)
metadata["Filename"] = metadata["Filename"].str[:5] + "." + metadata["Filename"].str[5:]

# ======================
# PROCESS
# ======================
features = []

for file in tqdm(metadata["Filename"].unique()):

    path = os.path.join(audio_dir, file)
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    subset = metadata[metadata["Filename"] == file].reset_index(drop=True)

    for i in range(len(subset)):

        start = max(0, int(sr * subset.loc[i, "Begin_Time__s_"] - sr * BUFFER_SEC))
        end = int(sr * subset.loc[i, "End_Time__s_"])

        segment = y[start:end]

        # Bandpass filter
        segment = butter_bandpass_filter(
            segment,
            subset.loc[i, "Low_Freq__Hz_"],
            subset.loc[i, "High_Freq__Hz_"],
            sr
        )

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            fmax=FMAX
        )

        # summarize
        feat = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])
        features.append(feat)

# ======================
# SAVE OUTPUT
# ======================
features = np.array(features)

np.save("results/mfcc_features.npy", features)
pd.DataFrame(features).to_csv("results/mfcc_features.csv", index=False)

print("MFCC extraction complete.")
