import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import wavfile
import librosa
from python_speech_features import mfcc, logfbank 
import warnings 

# Checking out how I can plot data 

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5)),

    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# Fast Fourier Transform 
def plot_fft(fft):
    fig, axes= plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))

    fig.suptitle('Fourier Transform', size=16)
    i = 0 
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle("Filter Bank Coefficients", size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                             cmap='hot', interpolations='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# Mel frequency cepstral coefficients 
def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=False, figsize=(20,5))
    
    fig.suptitle("Mel Frequency Cepstral Coefficients", size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                             cmap='hot', interpolations='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)



def process_dataset(csv_path, wav_folder_healthy, output_path):
    """
    Process lung sound dataset by reading wav files and calculating their lengths.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file with lung dataset
    wav_folder_healthy : str
        Path to the folder containing healthy wav files
    wav_folder_unhealthy : str, optional
        Path to the folder containing unhealthy wav files
    output_path : str, optional
        Path where to save the output CSV file
    
    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with audio lengths
    dict
        Dictionary with statistics about processing
    """
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded dataset with {len(df)} records")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
    
    # Set subject number as index if it exists
    if 'SUBJ. NO' in df.columns:
        df.set_index('SUBJ. NO', inplace=True)
        print("Set 'SUBJ. NO' as index")
    
    # Add length column if it doesn't exist
    if 'length' not in df.columns:
        df['length'] = np.nan
    
    # Statistics dictionary
    stats = {
        'total_files': len(df),
        'processed_files': 0,
        'missing_files': [],
        'error_files': []
    }
    
    # Process healthy files
    if wav_folder_healthy:
        stats_healthy = process_wav_files(df, wav_folder_healthy, 'h')
        stats['processed_files'] += stats_healthy['processed']
        stats['missing_files'].extend(stats_healthy['missing'])
        stats['error_files'].extend(stats_healthy['errors'])

    # Save updated DataFrame if output path is provided
    if output_path:
        try:
            df.to_csv(output_path)
            print(f"Updated DataFrame saved to {output_path}")
        except Exception as e:
            print(f"Error saving DataFrame: {e}")
    
    # Print processing summary
    print(f"\nProcessing summary:")
    print(f"  Total files in dataset: {stats['total_files']}")
    print(f"  Successfully processed: {stats['processed_files']}")
    print(f"  Missing files: {len(stats['missing_files'])}")
    print(f"  Files with errors: {len(stats['error_files'])}")
    
    # Save missing files list
    if stats['missing_files']:
        try:
            with open("missing_files.txt", "w") as f:
                for file in stats['missing_files']:
                    f.write(file + "\n")
            print(f"List of missing files saved to 'missing_files.txt'")
        except Exception as e:
            print(f"Error saving missing files list: {e}")
    
    return df, stats


def process_wav_files(df, wav_folder, file_suffix):
    """
    Process wav files in a folder and update DataFrame with lengths.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to update
    wav_folder : str
        Folder containing wav files
    file_suffix : str
        Suffix for file names ('h' for healthy, 'u' for unhealthy)
    
    Returns:
    --------
    dict
        Statistics about processing
    """
    results = {'processed': 0, 'missing': [], 'errors': []}
    
    print(f"\nProcessing files in {wav_folder} with suffix '{file_suffix}'...")
    
    # Ensure the wav folder exists
    if not os.path.exists(wav_folder):
        print(f"Warning: Folder {wav_folder} does not exist")
        return results
    
    # Loop through subjects in the dataset
    for subject_id in df.index:
        file_name = f'{subject_id}- {file_suffix}.wav'
        file_path = os.path.join(wav_folder, file_name)
        
        # Handle file processing
        if os.path.exists(file_path):
            try:
                rate, signal = wavfile.read(file_path)
                
                # Calculate length in seconds
                duration = signal.shape[0] / rate
                
                # Also update the general 'length' column for backward compatibility
                df.at[subject_id, 'length'] = duration
                
                results['processed'] += 1
                
                # Print progress every 10 files
                if results['processed'] % 10 == 0:
                    print(f"  Processed {results['processed']} files so far...")
                
            except Exception as e:
                error_msg = f"Error reading {file_path}: {e}"
                print(f"  {error_msg}")
                results['errors'].append((file_name, str(e)))
        else:
            results['missing'].append(file_name)
    
    print(f"Done processing {file_suffix} files: {results['processed']} processed, "
          f"{len(results['missing'])} missing, {len(results['errors'])} errors")
    
    return results


if __name__ == "__main__":
    # Configuration
    csv_path = "C:/Users/richa/Downloads/lung_dataset_healthy.csv"
    wav_folder_healthy = "wavfiles/healthy" 
    output_path = "C:/Users/richa/Downloads/lung_dataset_healthy_table.csv"
    
    # Process the dataset
    updated_df, stats = process_dataset(
        csv_path=csv_path,
        wav_folder_healthy=wav_folder_healthy,
        output_path=output_path
    )

# ============== Next ==========
# Processed the lengths of the audio/wavfiles in the csv 
# Main issue is that they have duplicates of the audio lengths
# Structured like ==== length, length_healthy === even for the unhealthy table
# Remove the unhealthy column with the times it has because I don't need duplicates of the audio lengths 