import os
import re
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.stats import kurtosis, skew

# -----------------------
# Config
# -----------------------
AUDIO_DIR = r"C:\Users\richa\OneDrive\Desktop\science2\data\wavfiles\unhealthy\38-"
OUTPUT_FILE = r"C:\Users\richa\OneDrive\Desktop\science2\data\jitter_shimmerlog.csv"

SAMPLE_RATE = 22050
CHUNK_DURATION = 2.0  
CANCER_STAGE = 1

def extract_features(y, sr):
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)

    # Spectral features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))

    # Higher-order stats
    crest_factor = np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-6)
    krt = kurtosis(y)
    skw = skew(y)

    # Combine
    features = {
        "zcr": zcr,
        "centroid": centroid,
        "rolloff": rolloff,
        "bandwidth": bandwidth,
        "rms": rms,
        "crest_factor": crest_factor,
        "kurtosis": krt,
        "skew": skw
    }

    # Add MFCCs (mean only)
    for i, m in enumerate(mfcc_mean):
        features[f"mfcc{i+1}_mean"] = m

    return features

# -----------------------
# Chunk Audio
# -----------------------
def chunk_audio(y, sr, chunk_duration=1.0):
    chunk_size = int(chunk_duration * sr)
    chunks = []
    for start in range(0, len(y), chunk_size):
        end = start + chunk_size
        seg = y[start:end]
        if len(seg) < chunk_size:
            seg = np.pad(seg, (0, chunk_size - len(seg)))  # pad last segment
        chunks.append(seg)
    return chunks

# -----------------------
# Main processing
# -----------------------
def process_folder(audio_dir, output_file, cancer_stage):
    rows = []

    for fname in os.listdir(audio_dir):
        if not fname.lower().endswith(".wav"):
            continue

        patient_id = os.path.splitext(fname.strip())[0]  # use filename as patient ID

        fpath = os.path.join(audio_dir, fname)
        y, sr = librosa.load(fpath, sr=SAMPLE_RATE)

        # Split into non-overlapping chunks
        chunks = chunk_audio(y, sr, CHUNK_DURATION)

        for i, chunk in enumerate(chunks):
            feats = extract_features(chunk, sr)
            feats.update({
                "patient_id": patient_id,
                "stage": cancer_stage,
                "chunk": i,
                "filename": fname
            })
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
    print(f"✅ Features saved to {output_file} with shape {df.shape}")
    return df

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    df = process_folder(AUDIO_DIR, OUTPUT_FILE, CANCER_STAGE)
    print(df.head())