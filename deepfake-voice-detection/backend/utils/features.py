import librosa
import numpy as np
import torch

def extract_log_mel_spectrogram(audio_path_or_array, sr=16000, duration=4.0):
    """
    Converts audio to a Log-Mel Spectrogram image tensor.
    Works for both file paths (Training) and raw bytes (Live).
    """
    target_len = int(sr * duration)
    
    # 1. Load Audio
    if isinstance(audio_path_or_array, str):
        # Load from file (Training)
        try:
            y, _ = librosa.load(audio_path_or_array, sr=sr)
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return torch.zeros(1, 128, 128) # Return dummy on error
    else:
        # Load from numpy array (Live)
        y = audio_path_or_array 
        
    # 2. Pad or Truncate to fixed length (4s)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # 3. Extract Mel Spectrogram
    # n_mels=128 is the height of the image
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256
    )
    
    # 4. Convert to Log Scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 5. Normalize (Crucial for Neural Nets to converge fast)
    mean = np.mean(log_mel)
    std = np.std(log_mel)
    log_mel = (log_mel - mean) / (std + 1e-6)
    
    # 6. Convert to Tensor [1, H, W]
    # We add the '1' channel dimension because CNNs expect (Channel, Height, Width)
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)