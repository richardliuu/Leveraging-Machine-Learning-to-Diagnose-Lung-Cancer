import os 
import pydub
import soundfile as sf 
import numpy as np
import librosa
from pydub import AudioSegment 
import noisereduce as nr 
import scipy as sk

# To extract features from the audio before segmenting them 

# Sampling rate
sr = 16000

def load_audio(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr 

def noise_reduce(y, sr):
    return nr.reduce_noise(y, sr)

def apply_limiter(audio_segment, threshold_dBFS=-2.0, release_ms=50):
    return audio_segment.apply_gain_and_limiter(threshold_dBFS, release_ms)

def normalize(audio_segment, target_dBFS=-20):
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def save_wav_file(y, sr, out_path):
    sf.write(out_path, y, sr, format='WAV')

def apply_compression(audio_segment, threshold_dB=-24, ratio=2.0):
    # Digital Signal Processing
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0 

# Above normalizes to [-1, 1] which ensures consistent loudness levels  
# Prevents clipping

    db_level = 20 * np.log10(np.abs(samples) + 1e-10)

    mask = db_level > threshold_dB
    gain_reduction = (db_level[mask] - threshold_dB * (1 - 1/ratio))

    samples[mask] *= 10**(-gain_reduction/20)

    samples = (samples * 32768.0).astype(np.int16)

    return AudioSegment(
        samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=2,
        channels=1
    )

def enhance_male_voice(y, sr):
    # Need a filter tha enhances the frequencies of male voices

    b1, a1 = sk.signal.butter(2, [400/(sr/2), 700/(sr/2)], btype='bandpass')
    y1 = sk.signal.lfilter(b1, a1, y)

    b2, a2 = sk.signal.butter(2, [1300/(sr/2), 1800/(sr/2)], btype='bandpass')
    y2= sk.signal.lfilter(b2, a2, y)

    # The enhanced voice 
    y_enhanced = y + 0.15 * y1 + 0.1 * y2

    return y_enhanced / np.max(np.abs(y_enhanced))

def preprocess_audio(input_path, output_path, target_sr=16000, target_dBFS=-20):
    y, sr = load_audio(input_path, target_sr)

    y_reduced_noise = noise_reduce(y, sr)

    y_highpassed = librosa.effects.hpss(y, margin=3.0)[0]

    # To save the result temporarily for further processing
    temp_path='temp.wav'
    save_wav_file(y_reduced_noise, sr, temp_path)

    audio_segment = AudioSegment.from_wav(temp_path)

    compression = apply_compression(audio_segment)

    limited = apply_limiter(compression)

    normalized = normalize(audio_segment, target_dBFS)
    normalized.export(output_path, format='wav')    

    os.remove(temp_path)

# Insert the name of the wav file 
input_file=r'C:\Users\richa\OneDrive\Desktop\science2\wavfiles\healthy\1- h.wav'

# Returns the cleaned version of it 
output_folder=r'C:\Users\richa\OneDrive\Desktop\science2\cleaned_wavfiles'
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "cleaned_1 - h.wav")

preprocess_audio(input_file, output_file)