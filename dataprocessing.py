import os
import numpy as np
import librosa
import scipy.signal
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import librosa.display

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

def visualize_results(input_path, output_path, plot_path=None):
    y_orig, sr_orig = librosa.load(input_path, sr=None)
    y_proc, sr_proc = librosa.load(output_path, sr=None)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig, n_fft=2048)), ref=np.max)
    librosa.display.specshow(D_orig, sr=sr_orig, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original')

    plt.subplot(2, 1, 2)
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc, n_fft=2048)), ref=np.max)
    librosa.display.specshow(D_proc, sr=sr_proc, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Minimally Processed Voice (Preserving Clinical Features)')

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path)
        print(f"Saved comparison plot to: {plot_path}")
    else:
        plt.show()

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

if __name__ == "__main__":
    input_file = r"C:\Users\richa\OneDrive\Desktop\science2\wavfiles\healthy\1- h.wav"
    output_file = r"C:\Users\richa\OneDrive\Desktop\science2\cleaned_wavfiles\extracted_1- h.wav"

    feature_comparison = minimal_process_voice(input_file, output_file, apply_noise_reduction=True)
    visualize_results(input_file, output_file, "minimal_processing_comparison.png")

    detailed_features = extract_clinical_voice_features(input_file)
    print("\nDetailed clinical voice features for ML:")
    for feature, value in detailed_features.items():
        print(f"{feature}: {value}")
