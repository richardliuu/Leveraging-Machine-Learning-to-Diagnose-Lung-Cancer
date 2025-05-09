from pydub import AudioSegment
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load audio using PyDub
file_path = 
audio = AudioSegment.from_file(file_path)

# Convert to mono and export as raw samples
samples = np.array(audio.get_array_of_samples()).astype(np.float32)
if audio.channels == 2:
    samples = samples.reshape((-1, 2)).mean(axis=1)  # stereo to mono

# Normalize to [-1, 1]
samples /= (2 ** (8 * audio.sample_width - 1))

# --- Analysis ---
duration_sec = len(samples) / audio.frame_rate
rms = np.sqrt(np.mean(samples**2))
peak = np.max(np.abs(samples))
rms_dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf
clipped_samples = np.sum(np.abs(samples) >= 1.0)

print(f"Duration: {duration_sec:.2f} sec")
print(f"RMS: {rms:.4f} ({rms_dbfs:.2f} dBFS)")
print(f"Peak: {peak:.4f} ({20 * np.log10(peak):.2f} dBFS)")
print(f"Clipped samples: {clipped_samples}")

# --- Plot Spectrogram ---
y, sr = librosa.load(file_path, sr=None)
plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
