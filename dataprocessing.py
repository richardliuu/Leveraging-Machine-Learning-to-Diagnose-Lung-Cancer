import os
import numpy as np
import librosa
import scipy.signal
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import librosa.display
import csv

def load_audio(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr

def save_wav_file(y, sr, out_path):
    sf.write(out_path, y, sr, format='WAV')

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
    low_normalized = max(0.001, min(low_freq / nyquist, 0.99))
    high_normalized = max(0.001, min(high_freq / nyquist, 0.99))
    b, a = scipy.signal.butter(2, [low_normalized, high_normalized], btype='bandpass')
    filtered = scipy.signal.filtfilt(b, a, y)
    return filtered

def analyze_audio_features(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic
    harmonic_rms = np.sqrt(np.mean(harmonic**2))
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-10
    hnr_estimate = 20 * np.log10(harmonic_rms / noise_rms)

    return {
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),
        "centroid_mean": np.mean(centroid),
        "centroid_std": np.std(centroid),
        "flatness_mean": np.mean(flatness),
        "flatness_std": np.std(flatness),
        "hnr_estimate": hnr_estimate
    }

def minimal_process_voice(input_path, output_path, apply_noise_reduction=True):
    print(f"Loading audio from: {input_path}")
    y, sr = load_audio(input_path, target_sr=16000)
    print(f"Processing audio with sample rate: {sr} Hz")
    original_features = analyze_audio_features(y, sr)
    print(f"Original audio features: {original_features}")

    y_filtered = apply_bandpass_filter(y, sr, 80, 8000)
    y_processed = apply_minimal_noise_reduction(y_filtered, sr) if apply_noise_reduction else y_filtered

    processed_features = analyze_audio_features(y_processed, sr)
    print(f"Processed audio features: {processed_features}")
    save_wav_file(y_processed, sr, output_path)
    print(f"Saved processed audio to: {output_path}")

    return {"original": original_features, "processed": processed_features}

def visualize_results(input_path, output_path, plot_folder="wavfiles_pngs"):
    os.makedirs(plot_folder, exist_ok=True)

    # Load the processed audio
    y_proc, sr_proc = librosa.load(output_path, sr=16000)

    # Create the spectrogram
    plt.figure(figsize=(6, 4))
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc, n_fft=2048)), ref=np.max)
    librosa.display.specshow(D_proc, sr=sr_proc, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Processed Voice Spectrogram')

    # Save to the designated folder
    filename = os.path.basename(output_path).replace(".wav", "_spectrogram.png")
    plot_path = os.path.join(plot_folder, filename)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved processed spectrogram to: {plot_path}")
    plt.close()

def extract_clinical_voice_features(input_path):
    y, sr = load_audio(input_path, target_sr=16000)
    features = analyze_audio_features(y, sr)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = [pitches[magnitudes[:, i].argmax(), i] for i in range(pitches.shape[1]) if pitches[magnitudes[:, i].argmax(), i] > 0]
    
    if pitch_values:
        pitch_diffs = np.abs(np.diff(pitch_values))
        jitter = np.mean(pitch_diffs) / np.mean(pitch_values) if np.mean(pitch_values) > 0 else 0
        features["jitter_approx"] = jitter
    else:
        features["jitter_approx"] = 0

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_std"] = np.std(mfccs[i])

    return features

def segment_audio(y, sr, segment_duration=3):
    """Split audio into segments of a specified duration in seconds."""
    segment_length = int(sr * segment_duration)
    total_length = len(y)
    segments = []

    for start in range(0, total_length, segment_length):
        end = min(start + segment_length, total_length)
        segment = y[start:end]
        if len(segment) >= int(0.5 * segment_length):  # keep at least 50% filled segments
            segments.append(segment)
        else:
            print(f"Skipping short segment: {len(segment)/sr:.2f}s")

    return segments

def segment_and_process_audio(input_path, output_folder, segment_duration=3, csv_path="features_log.csv"):
    """Split audio into segments, process each one, save it, log features to CSV, and visualize."""
    os.makedirs(output_folder, exist_ok=True)
    y, sr = load_audio(input_path, target_sr=16000)
    segments = segment_audio(y, sr, segment_duration)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    feature_list = []

    # CSV Header
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["segment", "rms_mean", "rms_std", "zcr_mean", "zcr_std", 
                                                      "centroid_mean", "centroid_std", "flatness_mean", "flatness_std", 
                                                      "hnr_estimate", "jitter_approx"] + [f"mfcc{i+1}_mean" for i in range(13)] + [f"mfcc{i+1}_std" for i in range(13)])
            writer.writeheader()

    for i, segment in enumerate(segments):
        segment_filename = f"{base_name}_seg{i+1}.wav"
        segment_path = os.path.join(output_folder, segment_filename)

        save_wav_file(segment, sr, segment_path)

        processed_segment_path = os.path.join(output_folder, f"{base_name}_seg{i+1}_processed.wav")

        print(f"\nProcessing segment {i+1}: {segment_path}")
        minimal_process_voice(segment_path, processed_segment_path, apply_noise_reduction=True)

        # Visualize processed audio
        visualize_results(segment_path, processed_segment_path, plot_folder=output_folder)

        # Extract and log features
        features = extract_clinical_voice_features(segment_path)
        features["segment"] = segment_filename  # Add segment info for identification

        # Append to list for CSV logging
        feature_list.append(features)

    # Write features to CSV
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=feature_list[0].keys())
        writer.writerows(feature_list)

    print(f"Logged features to {csv_path}")

if __name__ == "__main__":
    input_file = r"C:\Users\richa\OneDrive\Desktop\science2\wavfiles\healthy\1- h.wav"
    output_file = r"C:\Users\richa\OneDrive\Desktop\science2\cleaned_wavfiles\extracted_1- h.wav"
    segmented_output_dir = r"C:\Users\richa\OneDrive\Desktop\science2\segmented_processed_1"
    csv_log_path = r"C:\Users\richa\OneDrive\Desktop\science2\voice_features_log.csv"

    feature_comparison = minimal_process_voice(input_file, output_file, apply_noise_reduction=True)

    segment_and_process_audio(input_file, segmented_output_dir, segment_duration=3, csv_path=csv_log_path)

    detailed_features = extract_clinical_voice_features(input_file)
    print("\nDetailed clinical voice features for ML:")
    for feature, value in detailed_features.items():
        print(f"{feature}: {value}")
