"""
utils.py — Shared constants, configs, seed setup, and utility functions.
Used across all notebooks and scripts.
"""

import os
import random
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import wandb

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Constants
# ============================================================
SAMPLE_RATE = 16000
DURATION = 10            # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION
NUM_CLASSES = 10
DATASET_MULTIPLIER = 3   # 3x more samples per epoch

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}

# TTA settings (systematic overlapping windows)
TTA_WINDOW_STARTS_SEC = [0, 5, 10, 15, 20]
TTA_CROPS = len(TTA_WINDOW_STARTS_SEC)

# Ensemble weights
W_CNN = 0.15
W_CRNN = 0.20
W_HUBERT = 0.65

# ============================================================
# Path Config
# ============================================================
def get_paths():
    """Auto-detect Kaggle paths and return a dict of directories."""
    BASE_DIR = '/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup'
    if not os.path.exists(BASE_DIR):
        BASE_DIR = '/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup'

    STEMS_DIR = os.path.join(BASE_DIR, 'genres_stems')
    MASHUPS_DIR = os.path.join(BASE_DIR, 'mashups')

    if os.path.exists(os.path.join(BASE_DIR, 'ESC-50-master', 'audio')):
        NOISE_DIR = os.path.join(BASE_DIR, 'ESC-50-master', 'audio')
    elif os.path.exists(os.path.join(BASE_DIR, 'ESC-50', 'audio')):
        NOISE_DIR = os.path.join(BASE_DIR, 'ESC-50', 'audio')
    else:
        NOISE_DIR = None

    return {
        'BASE_DIR': BASE_DIR,
        'STEMS_DIR': STEMS_DIR,
        'MASHUPS_DIR': MASHUPS_DIR,
        'NOISE_DIR': NOISE_DIR,
        'TEST_CSV': os.path.join(BASE_DIR, 'test.csv'),
        'SAMPLE_SUB': os.path.join(BASE_DIR, 'sample_submission.csv'),
    }

# ============================================================
# WandB Login
# ============================================================
def wandb_login():
    """Login to WandB using Kaggle Secrets or environment variable."""
    WANDB_KEY = None
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        WANDB_KEY = user_secrets.get_secret("WANDB_API_KEY")
        print("Got WandB key from Kaggle Secrets.")
    except Exception:
        print("Kaggle Secrets not available. Set WANDB_API_KEY env var or login manually.")

    if WANDB_KEY:
        try:
            os.environ["WANDB_API_KEY"] = WANDB_KEY
            wandb.login()
            print("WandB logged in successfully.")
        except Exception as e:
            print(f"WandB login failed: {e}")
            os.environ["WANDB_MODE"] = "offline"
    else:
        print("No WandB key found. Running in offline mode.")
        os.environ["WANDB_MODE"] = "offline"

# ============================================================
# Audio Utilities
# ============================================================
def mix_stems_to_audio(song_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load and mix all 4 stems from a song folder into one audio array."""
    num_frames = int(sr * duration)
    mixed = np.zeros(num_frames)
    for stem in ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']:
        stem_path = os.path.join(song_path, stem)
        if os.path.exists(stem_path):
            y, _ = librosa.load(stem_path, sr=sr, duration=duration)
            if len(y) < num_frames:
                y = np.pad(y, (0, num_frames - len(y)))
            else:
                y = y[:num_frames]
            mixed += y
    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val
    return mixed


def extract_features_from_array(y, sr=SAMPLE_RATE):
    """Extract a rich feature vector (MFCC + Chroma + Spectral) from audio array."""
    features = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))))
    features.append(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))))
    features.append(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))))
    features.append(float(np.mean(librosa.feature.zero_crossing_rate(y=y))))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(np.asarray(tempo).item()))
    return np.array(features)


def extract_features_from_file(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Extract features from a single audio file."""
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    num_frames = int(sr * duration)
    if len(y) < num_frames:
        y = np.pad(y, (0, num_frames - len(y)))
    else:
        y = y[:num_frames]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return extract_features_from_array(y, sr)


def build_file_lookup(mashups_dir):
    """Build a lookup dict mapping various naming conventions to file paths."""
    available_files = {}
    for f in os.listdir(mashups_dir):
        full_path = os.path.join(mashups_dir, f)
        available_files[f] = full_path
        name_no_ext = os.path.splitext(f)[0]
        available_files[name_no_ext] = full_path
    return available_files


def find_test_file(file_id, available_files, row=None):
    """Try multiple naming conventions to find a test file."""
    import pandas as pd
    file_id = str(file_id)
    candidates = []
    if row is not None and 'filename' in row.index and pd.notna(row.get('filename')):
        candidates.append(os.path.basename(str(row['filename'])))
    try:
        int_id = int(file_id)
        candidates.extend([f"song{int_id:04d}.wav", f"song{int_id:04d}"])
    except ValueError:
        pass
    candidates.extend([f"{file_id}.wav", file_id, f"{file_id}.mp3"])
    try:
        int_id = int(file_id)
        candidates.extend([f"{int_id:04d}.wav", f"{int_id:04d}",
                           f"{int_id:03d}.wav", f"{int_id:05d}.wav"])
    except ValueError:
        pass
    for c in candidates:
        if c in available_files:
            return available_files[c]
    return None


def load_systematic_tta_crops(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load full audio and extract systematic overlapping windows for TTA.
    Windows: [0-10s, 5-15s, 10-20s, 15-25s, 20-30s]
    """
    waveform, file_sr = torchaudio.load(file_path)
    if file_sr != sr:
        waveform = T.Resample(file_sr, sr)(waveform)
    waveform = torch.mean(waveform, dim=0)  # mono

    num_samples = sr * duration
    total_len = waveform.shape[0]

    crops = []
    for start_sec in TTA_WINDOW_STARTS_SEC:
        start_sample = int(start_sec * sr)
        end_sample = start_sample + num_samples

        if end_sample <= total_len:
            crop = waveform[start_sample:end_sample]
        elif start_sample < total_len:
            crop = waveform[start_sample:]
            crop = torch.nn.functional.pad(crop, (0, num_samples - crop.shape[0]))
        else:
            if total_len >= num_samples:
                crop = waveform[total_len - num_samples:]
            else:
                crop = torch.nn.functional.pad(waveform, (0, num_samples - total_len))

        crop = crop / (torch.max(torch.abs(crop)) + 1e-8)
        crops.append(crop)

    return torch.stack(crops)  # (TTA_CROPS, num_samples)


def cleanup_memory(*objects):
    """Delete objects and free GPU memory."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleaned up.")
