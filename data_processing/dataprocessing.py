import os
import numpy as np
import librosa
import scipy.signal
import noisereduce as nr
import csv
import parselmouth
import pandas as pd
import soundfile as sf

def load_audio(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr

def apply_minimal_noise_reduction(y, sr, reduction_strength=0.3):
    noise_sample = y[:int(sr * 0.5)] if len(y) > sr * 0.5 else y[:int(len(y) * 0.1)]
    return nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=True,
        prop_decrease=reduction_strength,
        n_fft=2048,
        win_length=1024,
        n_std_thresh_stationary=1.5
    )

def apply_bandpass_filter(y, sr, low_freq=80, high_freq=8000):
    nyquist = sr / 2
    low = max(0.001, min(low_freq / nyquist, 0.99))
    high = max(0.001, min(high_freq / nyquist, 0.99))
    b, a = scipy.signal.butter(2, [low, high], btype='bandpass')
    return scipy.signal.filtfilt(b, a, y)

def pad_to_fixed_length(y, sr, duration=3):
    target_len = int(sr * duration)
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    else:
        return y[:target_len]

def analyze_audio_features(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic
    h_rms = np.sqrt(np.mean(harmonic**2))
    n_rms = np.sqrt(np.mean(noise**2)) + 1e-10
    hnr = 20 * np.log10(h_rms / n_rms)

    return {
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),
        "centroid_mean": np.mean(centroid),
        "centroid_std": np.std(centroid),
        "flatness_mean": np.mean(flatness),
        "flatness_std": np.std(flatness),
        "hnr_estimate": hnr
    }

def extract_parselmouth_features(path):
    try:
        snd = parselmouth.Sound(path)
        pitch = parselmouth.praat.call(snd, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) == 0:
            pitch_mean = 0
            pitch_std = 0
        else:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)

        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std
        }
    except Exception as e:
        print(f"Error in parselmouth analysis: {e}")
        return {
            "pitch_mean": 0,
            "pitch_std": 0
        }

def extract_all_features(y, sr, path=None):
    features = analyze_audio_features(y, sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_std"] = np.std(mfccs[i])
    if path:
        features.update(extract_parselmouth_features(path))
    return features, mfccs

def fix_nan_values(features_list):
    df = pd.DataFrame(features_list)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            mean_val = df[col].mean()
            replacement = 0 if np.isnan(mean_val) else mean_val
            df[col] = df[col].fillna(replacement)
    return df.to_dict('records')

def append_to_npy(new_data, npy_path):
    if os.path.exists(npy_path):
        existing = np.load(npy_path, allow_pickle=True)
        combined = list(existing) + new_data
    else:
        combined = new_data
    np.save(npy_path, np.array(combined, dtype=object), allow_pickle=True)

def process_and_segment_audio(input_path, output_folder, csv_path, npy_path, cancer_stage, patient_id, segment_duration=3):
    os.makedirs(output_folder, exist_ok=True)
    y, sr = load_audio(input_path)
    segment_len = int(sr * segment_duration)

    features_all = []
    header_fields = [
        "patient_id", "segment", "cancer_stage", "rms_mean", "rms_std", "zcr_mean", "zcr_std",
        "centroid_mean", "centroid_std", "flatness_mean", "flatness_std", "hnr_estimate",
        "pitch_mean", "pitch_std"
    ] + [f"mfcc{i+1}_mean" for i in range(13)] + [f"mfcc{i+1}_std" for i in range(13)]

    all_mfccs = []

    for i in range(0, len(y), segment_len):
        segment = y[i:i+segment_len]
        if len(segment) < 0.5 * segment_len:
            continue
        segment = pad_to_fixed_length(segment, sr, segment_duration)
        filtered = apply_bandpass_filter(segment, sr)
        reduced = apply_minimal_noise_reduction(filtered, sr)

        try:
            features, mfccs = extract_all_features(reduced, sr, input_path)
            features["segment"] = f"{os.path.splitext(os.path.basename(input_path))[0]}_seg{i//segment_len+1}"
            features["cancer_stage"] = cancer_stage
            features["patient_id"] = patient_id
            features_all.append(features)
            all_mfccs.append(mfccs.T)  # Transpose for time steps × MFCCs
        except Exception as e:
            print(f"Warning: Failed to extract features for segment {i//segment_len+1}: {e}")
            continue

    if features_all:
        features_all = fix_nan_values(features_all)
        df_new = pd.DataFrame(features_all)
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(csv_path, index=False)

    if all_mfccs:
        labeled_mfccs = [(mfcc, cancer_stage, patient_id) for mfcc in all_mfccs]
        append_to_npy(labeled_mfccs, npy_path)

    print(f"Finished processing {input_path}. Segments: {len(all_mfccs)}")

if __name__ == "__main__":
    input_file = r"C:\\Users\\richa\\OneDrive\\Desktop\\science2\\wavfiles\\unhealthy\\50- u.wav"
    output_dir = r"C:\\Users\\richa\\OneDrive\\Desktop\\science2\\segments_padded"
    csv_log = r"C:\\Users\\richa\\OneDrive\\Desktop\\science2\\voice_features_log.csv"
    mfcc_npy_file = r"C:\\Users\\richa\\OneDrive\\Desktop\\science2\\all_mfccs.npy"
    cancer_stage = 2  # 0=healthy, 1=stage1, 2=stage2, 3=stage3, 4=stage4
    patient_id = "U050"
    process_and_segment_audio(input_file, output_dir, csv_log, mfcc_npy_file, cancer_stage, patient_id)

