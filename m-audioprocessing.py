import os
import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
import soundfile as sf
from pydub import AudioSegment
import noisereduce as nr

def load_audio(path, target_sr=16000):
    """Load audio file with target sample rate"""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr

def save_wav_file(y, sr, out_path):
    """Save audio data to WAV file"""
    sf.write(out_path, y, sr, format='WAV')

def noise_reduce(y, sr):
    """Apply noise reduction"""
    return nr.reduce_noise(y, sr)

def normalize(audio_segment, target_dBFS=-22):
    """Normalize audio to target dB level"""
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def apply_limiter(audio_segment, threshold_dB=-1.0, release_ms=50):
    """Apply limiter to prevent clipping"""
    # Simple implementation
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0  # Normalize to [-1, 1]
    
    # Calculate level
    db_level = 20 * np.log10(np.abs(samples) + 1e-10)
    
    # Apply limiting
    threshold_linear = 10 ** (threshold_dB / 20)
    mask = np.abs(samples) > threshold_linear
    gain_reduction = threshold_linear / (np.abs(samples[mask]) + 1e-10)
    samples[mask] *= gain_reduction
    
    # Convert back
    samples = (samples * 32767).astype(np.int16)
    
    # Create new AudioSegment
    return AudioSegment(
        samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=2,
        channels=1
    )

def enhance_male_voice(y, sr):
    """Enhance typical male voice characteristics"""
    # First formant enhancement
    b1, a1 = scipy.signal.butter(2, [400/(sr/2), 700/(sr/2)], btype='bandpass')
    y1 = scipy.signal.lfilter(b1, a1, y)
    
    # Second formant enhancement (for clarity)
    b2, a2 = scipy.signal.butter(2, [1300/(sr/2), 1800/(sr/2)], btype='bandpass')
    y2 = scipy.signal.lfilter(b2, a2, y)
    
    # Mix with original (subtle enhancement)
    y_enhanced = y + 0.15 * y1 + 0.1 * y2
    
    # Normalize
    return y_enhanced / np.max(np.abs(y_enhanced))

def apply_simple_compression(samples, threshold_db, ratio):
    """Apply simple compression to samples"""
    # Calculate level in dB
    db_level = 20 * np.log10(np.abs(samples) + 1e-10)
    
    # Apply compression
    mask = db_level > threshold_db
    gain_reduction = (db_level[mask] - threshold_db) * (1 - 1/ratio)
    samples_comp = samples.copy()
    samples_comp[mask] *= 10**(-gain_reduction/20)
    
    return samples_comp

def apply_compression_male_voice(audio_segment):
    """Apply multiband compression optimized for male voice"""
    # Extract samples
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0  # Normalize to [-1, 1]
    
    # Split into frequency bands
    sr = audio_segment.frame_rate
    # Low band (fundamental and first formant: 80-700 Hz)
    b_low, a_low = scipy.signal.butter(2, 700/(sr/2), btype='low')
    y_low = scipy.signal.lfilter(b_low, a_low, samples)
    
    # Mid band (mid frequencies: 700-2500 Hz)
    b_mid, a_mid = scipy.signal.butter(2, [700/(sr/2), 2500/(sr/2)], btype='bandpass')
    y_mid = scipy.signal.lfilter(b_mid, a_mid, samples)
    
    # High band (consonants and clarity: >2500 Hz)
    b_high, a_high = scipy.signal.butter(2, 2500/(sr/2), btype='high')
    y_high = scipy.signal.lfilter(b_high, a_high, samples)
    
    # Apply different compression settings to each band
    # Low band: gentle compression to preserve warmth
    y_low_comp = apply_simple_compression(y_low, threshold_db=-24, ratio=1.5)
    
    # Mid band: medium compression for consistency
    y_mid_comp = apply_simple_compression(y_mid, threshold_db=-20, ratio=2.5)
    
    # High band: stronger compression for controlled brightness
    y_high_comp = apply_simple_compression(y_high, threshold_db=-30, ratio=3.0)
    
    # Recombine with slight emphasis on midrange for male voice intelligibility
    y_processed = y_low_comp + 1.2 * y_mid_comp + 0.9 * y_high_comp
    
    # Normalize
    y_processed /= np.max(np.abs(y_processed))
    y_processed = (y_processed * 32767).astype(np.int16)
    
    # Create new AudioSegment
    return AudioSegment(
        y_processed.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=2,
        channels=1
    )

def apply_deesser(y, sr):
    """Apply de-essing to control sibilance in male voice"""
    # Focus on 4-8 kHz range where sibilance occurs
    b_sibilance, a_sibilance = scipy.signal.butter(2, [4000/(sr/2), 8000/(sr/2)], btype='bandpass')
    sibilance = scipy.signal.lfilter(b_sibilance, a_sibilance, y)
    
    # Dynamic compression on sibilance band
    sibilance_env = np.abs(sibilance)
    sibilance_smooth = scipy.ndimage.gaussian_filter1d(sibilance_env, sigma=sr*0.003)
    
    # Create gain reduction when sibilance is too strong
    threshold = 0.1 * np.max(sibilance_smooth)
    gain_reduction = np.ones_like(sibilance_smooth)
    mask = sibilance_smooth > threshold
    gain_reduction[mask] = threshold / (sibilance_smooth[mask] + 1e-10)
    
    # Apply to sibilance band only
    reduced_sibilance = sibilance * gain_reduction
    
    # Mix back
    y_deessed = y - sibilance + reduced_sibilance
    
    return y_deessed

def add_male_vocal_presence(y, sr):
    """Add presence to make male voice more forward in the mix"""
    # Gentle saturation to add harmonics
    def soft_clip(x, amount=0.1):
        return np.tanh(amount * x) / np.tanh(amount)
    
    y_saturated = soft_clip(y, amount=2.0)
    
    # Presence boost around 3-5 kHz
    b_presence, a_presence = scipy.signal.butter(2, [3000/(sr/2), 5000/(sr/2)], btype='bandpass')
    y_presence = scipy.signal.lfilter(b_presence, a_presence, y)
    
    # Mix with original
    y_enhanced = y + 0.05 * y_saturated + 0.2 * y_presence
    
    # Normalize
    return y_enhanced / np.max(np.abs(y_enhanced))

def validate_output(output_path):
    """Check the processed file for issues"""
    y, sr = librosa.load(output_path, sr=None)
    
    # Check for clipping
    clip_count = np.sum(np.abs(y) >= 0.99)
    
    # Check frequency spectrum
    D = np.abs(librosa.stft(y))
    freq_profile = np.mean(D, axis=1)
    high_freq_energy = np.sum(freq_profile[int(len(freq_profile)/2):])
    low_freq_energy = np.sum(freq_profile[:int(len(freq_profile)/2)])
    hf_lf_ratio = high_freq_energy / (low_freq_energy + 1e-10)
    
    return {
        "clipping": clip_count,
        "hf_lf_ratio": hf_lf_ratio,
        "rms_level": 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)
    }

def process_male_voice(input_path, output_path):
    """Complete processing chain for male voice"""
    # Load audio
    y, sr = load_audio(input_path, target_sr=16000)
    
    # Step 1: High-pass filter at 80Hz to remove rumble while preserving male voice
    b_hp, a_hp = scipy.signal.butter(2, 80/(sr/2), btype='high')
    y_highpassed = scipy.signal.filtfilt(b_hp, a_hp, y)
    
    # Step 2: Noise reduction
    y_reduced_noise = noise_reduce(y_highpassed, sr)
    
    # Step 3: Male voice formant enhancement
    y_enhanced = enhance_male_voice(y_reduced_noise, sr)
    
    # Step 4: Add vocal presence
    y_presence = add_male_vocal_presence(y_enhanced, sr)
    
    # Step 5: De-essing
    y_deessed = apply_deesser(y_presence, sr)
    
    # Convert to pydub AudioSegment for level processing
    temp_path = 'temp.wav'
    save_wav_file(y_deessed, sr, temp_path)
    audio_segment = AudioSegment.from_wav(temp_path)
    
    # Step 6: Apply multiband compression optimized for male voice
    compressed = apply_compression_male_voice(audio_segment)
    
    # Step 7: Apply limiter
    limited = apply_limiter(compressed, threshold_dB=-1.0)
    
    # Step 8: Normalize
    normalized = normalize(limited, target_dBFS=-22)
    
    # Export final result
    normalized.export(output_path, format='wav')
    
    # Cleanup
    os.remove(temp_path)
    
    # Validate result
    quality_metrics = validate_output(output_path)
    return quality_metrics

def visualize_results(input_path, output_path, plot_path=None):
    """Create before/after spectrograms for comparison"""
    import matplotlib.pyplot as plt
    
    # Load both files
    y_orig, sr_orig = librosa.load(input_path, sr=None)
    y_proc, sr_proc = librosa.load(output_path, sr=None)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Original spectrogram
    plt.subplot(2, 1, 1)
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
    librosa.display.specshow(D_orig, sr=sr_orig, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original')
    
    # Processed spectrogram
    plt.subplot(2, 1, 2)
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)
    librosa.display.specshow(D_proc, sr=sr_proc, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Processed Male Voice')
    
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path)
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Copy from computer path
    input_file = r"input_male_voice.wav"  
    output_file = r"processed_male_voice.wav"
    
    metrics = process_male_voice(input_file, output_file)
    print(f"Processing complete. Quality metrics: {metrics}")
    
    # Visualize results
    visualize_results(input_file, output_file, "before_after_comparison.png")