# Female Audio Processing 
import os 
import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import matplotlib.pyplot as plt

def load_audio(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr

def save_wav_file(y, sr, out_path):
    sf.write(out_path, y, sr, format='WAV')

def noise_reduce(y, sr):
    return nr.reduce_noise(y, sr)

# Higher dBFS because of the female voice 
def normalize(audio_segment, target_dBFS=-21):
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

# Limiter to prevent audio clipping
def apply_limiter(audio_segment, threshold_dB=-1.5, release_ms=50):
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0

    # For normalizing to [-1 , 1]

    db_level = 20 * np.log10(np.abs(samples) + 1e-10)

    threshold_linear = 10 ** (threshold_dB / 20)
    mask = np.abs(samples) > threshold_linear

    gain_reduction = threshold_linear / (np.abs(samples[mask] + 1e-10))
    samples[mask] *= gain_reduction

    samples = (samples * 32767).astype(np.int16)

    return AudioSegment(
        samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=2,
        channels=1
    )

def enhance_female_voice(y, sr):
    b1, a1 = scipy.signal.butter(2, [600/(sr/2), 900/(sr/2)], btype='bandpass')
    y1 = scipy.signal.lfilter(b1, a1, y)

    b2, a2 = scipy.signal.butter(2, [1700/(sr/2), 2200/(sr/2)], btype='bandpass')
    y2 = scipy.signal.lfilter(b2, a2, y)

    b3, a3 = scipy.signal.butter(2, [2800/(sr/2), 3500/(sr/2)], btype='bandpass')
    y3 = scipy.signal.lfilter(b3, a3, y)

    y_enhanced = y + 0.15 * y1 + 0.12 * y2 + 0.07 * y3 

    return y_enhanced/np.max(np.abs(y_enhanced))

def apply_simple_compression(samples, threshold_db, ratio):
    
    db_level = 20 * np.log10(np.abs(samples) + 1e-10)
    mask = db_level > threshold_db
    gain_reduction = (db_level[mask] - threshold_db) * (1 - 1/ratio)
    samples_comp = samples.copy()
    samples_comp[mask] *= 10**(-gain_reduction/20)

    return samples_comp 

def apply_compression_female_voice(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0 # For [-1, 1]

    sr = audio_segment.frame_rate

    # The first formant (150-900 hz)
    # Also the low band 
    b_low, a_low = scipy.signal.butter(2, 900/(sr/2), btype='low')
    y_low = scipy.signal.lfilter(b_low, a_low, samples)

    # Mid band which are frequencies 900-3000 hz
    b_mid, a_mid = scipy.signal.butter(2, [900/(sr/2), 3000/(sr/2)], btype='bandpass')
    y_mid = scipy.signal.lfilter(b_mid, a_mid, samples)

    # High band which is >3000hz and deals with consonants and clarity 
    b_high, a_high = scipy.signal.butter(2, 3000/(sr/2), btype='high')
    y_high = scipy.signal.lfilter(b_mid, a_mid, samples)

    # Applying compression to each band 
    # Controls dynamics
    y_low_comp = apply_simple_compression(y_low, threshold_db=-22, ratio=2.0)

    # Preserves natural tone 
    y_mid_comp = apply_simple_compression(y_mid, threshold_db=-20, ratio=1.8)

    # Brightness control
    y_high_comp = apply_simple_compression(y_high, threshold_db=-25, ratio=2.2)

    # Combination of all bands 
    y_processed = 0.9 * y_low_comp + 1.1 * y_mid_comp + y_high_comp

    y_processed /= np.max(np.abs(y_processed))

    return AudioSegment(
        y_processed.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=2,
        channels=1
    )

def apply_deesser_female(y, sr):
    """
    Look into sibliance 
    """
    b_sibliance, a_sibliance = scipy.signal.butter(2, [5000/(sr/2), 10000/(sr/2)], btype='bandpass')
    sibliance = scipy.signal.lfilter(b_sibliance, a_sibliance, y)

    sibliance_env = np.abs(sibliance)
    sibliance_smooth = scipy.ndimage.gaussian_filter1d(sibliance_env, sigma=sr*0.002)

    threshold = 0.08 * np.max(sibliance_smooth)
    gain_reduction = np.ones_like(sibliance_smooth)
    mask = sibliance_smooth > threshold
    gain_reduction[mask] = threshold / (sibliance_smooth[mask] + 1e-10)

    reduced_sibliance = sibliance * gain_reduction
    
    y_deessed = y - sibliance + reduced_sibliance
    
    return y_deessed

def add_female_vocal_air(y, sr):
    # Frequency enhancement 
    b_air, a_air = scipy.signal.butter(2, 7000/(sr/2), btype='high')
    y_air = scipy.signal.lfilter(b_air, a_air, y)

    def subtle_exciter(x, amount=0.2):
        return x + amount * (x - x**3/3)
    
    # Applying the exciter to mid-high frequencies

    b_mids, a_mids = scipy.signal.butter(2, [1500/(sr/2), 6000/(sr/2)], btype='bandpass')
    y_mids = scipy.signal.lfilter(b_mids, a_mids, y)
    y_mids_excited = subtle_exciter(y_mids, amount=0.15)

    y_with_mids = y - y_mids + y_mids_excited

    y_enhanced = y_with_mids + 0.12 * y_air

def validate_output(output_path):
    y, sr = librosa.load(output_path, sr=None)

    clip_count = np.sum(np.abs(y) >=0.99)

    D = np.abs(librosa.stft(y))
    freq_profile = np.mean(D, axis=1)
    high_freq_energy = np.sum(freq_profile[int(len(freq_profile)/2):])
    low_freq_energy = np.sum(freq_profile[:int(len(freq_profile)/2)])
    
    # High frequency and low frequency ratio
    hf_lf_ratio = high_freq_energy / (low_freq_energy + 1e-10)

    return {
        "clipping": clip_count,
        "hf_lf_ratio": hf_lf_ratio,
        "rms_level": 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)
    }

def process_female_voice(input_path, output_path):
    # Final part of the processing chain
    # Basically the pipeline

    y, sr = load_audio(input_path, target_sr = 16000)

    # High pass filter at 130 hz
    b_hp, a_hp = scipy.signal.butter(2, 130/(sr/2), btype='high')
    y_highpassed = scipy.signal.lfilter(b_hp, a_hp, y)

    # Noise reduction
    y_reduced_noise = noise_reduce(y_highpassed, sr)

    # Formant enhancement
    y_enhanced = enhance_female_voice(y_reduced_noise, sr)

    # Air and clarity
    y_with_air = add_female_vocal_air(y_enhanced, sr)

    # De-essing 
    y_deessed = apply_deesser_female(y_with_air, sr)

    # For pydub to process because it can't take numpy arrays 
    temp_path = 'temp.wav'
    save_wav_file(y_deessed, sr, temp_path)
    audio_segment = AudioSegment.from_wav(temp_path)

    # Multiband compression
    compressed = apply_compression_female_voice(audio_segment)

    # Apply limiter 
    limited = apply_limiter(compressed, threshold_dB=-1.5)

    # Normalization 
    normalized = normalize(limited, target_dBFS=-21)

    # Exporting final result 
    normalized.export(output_path, format='wav')

    os.remove(temp_path)

    quality_metrics = validate_output(output_path)

def visualize_results(input_path, output_path, plot_path=None):
    y_orig, sr_orig = librosa.load(input_path, sr=None)
    y_proc, sr_proc = librosa.load(output_path, sr=None)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig), ref=np.max))
    librosa.display.specshow(D_orig, sr=sr_orig, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original")

    plt.subplot(2, 1, 1)
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)
    librosa.display.specshow(D_proc, sr=sr_proc, x_axis='time', y_axis='log', cmap='magma')

    plt.tight_layout()

    if plot_path:
        plt.savefig(plot_path)
    else:
        plt.show()

if __name__ == "__main__":

    # Copy their paths 
    input_file = r"input_female_voice.wav"
    output_file = r"processed_female_voice.wav"

    metrics = process_female_voice(input_file, output_file)
    print(f"Processing complete. Quality Metrics: {metrics}")

    visualize_results(input_file, output_file, "female_b4a_comparison.png")