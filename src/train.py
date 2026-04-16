"""
train.py — Model definitions, dataset classes, and training functions.
Contains: GenreCNN, GenreCRNN, CrossSongMelDataset, CrossSongHubertDataset,
          and training loops for all milestones.
"""

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import HubertForSequenceClassification

import wandb
from tqdm import tqdm

from utils import (
    SEED, SAMPLE_RATE, DURATION, NUM_SAMPLES, NUM_CLASSES,
    DATASET_MULTIPLIER, GENRES, GENRE_TO_IDX, IDX_TO_GENRE,
    set_seed, get_paths, wandb_login,
    mix_stems_to_audio, extract_features_from_array,
    extract_features_from_file, build_file_lookup, find_test_file,
    cleanup_memory,
)


# ============================================================
# Dataset Classes
# ============================================================

class CrossSongMelDataset(Dataset):
    """Cross-song stem mixing dataset returning mel spectrograms (CNN/CRNN).
    Picks each stem from a DIFFERENT song of the same genre.
    Dataset multiplied 3x for more unique cross-song combinations per epoch.
    """
    def __init__(self, stems_dir, noise_dir, genres, duration=DURATION, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration
        self.stem_names = ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']
        self.genre_songs = {}
        self.all_samples = []
        for genre in genres:
            genre_path = os.path.join(stems_dir, genre)
            if os.path.exists(genre_path):
                song_paths = [os.path.join(genre_path, s) for s in os.listdir(genre_path)]
                self.genre_songs[genre] = song_paths
                for _ in song_paths:
                    self.all_samples.append(genre)
        base_len = len(self.all_samples)
        self.all_samples = self.all_samples * DATASET_MULTIPLIER
        self.noise_files = []
        if noise_dir and os.path.exists(noise_dir):
            self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64)
        self.amplitude_to_db = T.AmplitudeToDB()
        print(f"CrossSongMelDataset: {len(self.all_samples)} samples ({base_len} x {DATASET_MULTIPLIER}), {len(self.noise_files)} noise files")

    def __len__(self):
        return len(self.all_samples)

    def _load_stem(self, song_path, stem_name):
        stem_path = os.path.join(song_path, stem_name)
        if not os.path.exists(stem_path):
            return torch.zeros(1, self.num_samples)
        waveform, sr = torchaudio.load(stem_path)
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        total_len = waveform.shape[1]
        if total_len > self.num_samples:
            start = random.randint(0, total_len - self.num_samples)
            waveform = waveform[:, start:start + self.num_samples]
        elif total_len < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - total_len))
        return waveform

    def __getitem__(self, idx):
        genre = self.all_samples[idx]
        label = GENRE_TO_IDX[genre]
        songs_in_genre = self.genre_songs[genre]
        mixed = torch.zeros(1, self.num_samples)
        for stem_name in self.stem_names:
            random_song = random.choice(songs_in_genre)
            stem_audio = self._load_stem(random_song, stem_name)
            mixed += stem_audio * random.uniform(0.7, 1.3)
        if self.noise_files and random.random() < 0.7:
            noise_wf, sr = torchaudio.load(random.choice(self.noise_files))
            if sr != self.sample_rate:
                noise_wf = T.Resample(sr, self.sample_rate)(noise_wf)
            noise_wf = torch.mean(noise_wf, dim=0, keepdim=True)
            if noise_wf.shape[1] < self.num_samples:
                noise_wf = F.pad(noise_wf, (0, self.num_samples - noise_wf.shape[1]))
            else:
                noise_wf = noise_wf[:, :self.num_samples]
            signal_power = torch.mean(mixed ** 2) + 1e-10
            noise_power = torch.mean(noise_wf ** 2) + 1e-10
            snr_db = random.uniform(5, 20)
            scale = torch.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power))
            mixed = mixed + scale * noise_wf
        mixed = mixed / (torch.max(torch.abs(mixed)) + 1e-8)
        mel_spec = self.mel_transform(mixed)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db, label


class CrossSongHubertDataset(Dataset):
    """Cross-song stem mixing dataset returning raw 1D waveforms (HuBERT).
    Same mixing logic as CrossSongMelDataset. Dataset multiplied 3x.
    """
    def __init__(self, stems_dir, noise_dir, genres, duration=DURATION, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration
        self.stem_names = ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']
        self.genre_songs = {}
        self.all_samples = []
        for genre in genres:
            genre_path = os.path.join(stems_dir, genre)
            if os.path.exists(genre_path):
                song_paths = [os.path.join(genre_path, s) for s in os.listdir(genre_path)]
                self.genre_songs[genre] = song_paths
                for _ in song_paths:
                    self.all_samples.append(genre)
        base_len = len(self.all_samples)
        self.all_samples = self.all_samples * DATASET_MULTIPLIER
        self.noise_files = []
        if noise_dir and os.path.exists(noise_dir):
            self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
        print(f"CrossSongHubertDataset: {len(self.all_samples)} samples ({base_len} x {DATASET_MULTIPLIER})")

    def __len__(self):
        return len(self.all_samples)

    def _load_stem(self, song_path, stem_name):
        stem_path = os.path.join(song_path, stem_name)
        if not os.path.exists(stem_path):
            return torch.zeros(1, self.num_samples)
        waveform, sr = torchaudio.load(stem_path)
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        total_len = waveform.shape[1]
        if total_len > self.num_samples:
            start = random.randint(0, total_len - self.num_samples)
            waveform = waveform[:, start:start + self.num_samples]
        elif total_len < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - total_len))
        return waveform

    def __getitem__(self, idx):
        genre = self.all_samples[idx]
        label = GENRE_TO_IDX[genre]
        songs_in_genre = self.genre_songs[genre]
        mixed = torch.zeros(1, self.num_samples)
        for stem_name in self.stem_names:
            random_song = random.choice(songs_in_genre)
            stem_audio = self._load_stem(random_song, stem_name)
            gain = random.uniform(0.7, 1.3)
            mixed += stem_audio * gain
        if self.noise_files and random.random() < 0.7:
            noise_wf, sr = torchaudio.load(random.choice(self.noise_files))
            if sr != self.sample_rate:
                noise_wf = T.Resample(sr, self.sample_rate)(noise_wf)
            noise_wf = torch.mean(noise_wf, dim=0, keepdim=True)
            if noise_wf.shape[1] < self.num_samples:
                noise_wf = F.pad(noise_wf, (0, self.num_samples - noise_wf.shape[1]))
            else:
                noise_wf = noise_wf[:, :self.num_samples]
            signal_power = torch.mean(mixed ** 2) + 1e-10
            noise_power = torch.mean(noise_wf ** 2) + 1e-10
            snr_db = random.uniform(5, 20)
            scale = torch.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power))
            mixed = mixed + scale * noise_wf
        mixed = mixed / (torch.max(torch.abs(mixed)) + 1e-8)
        return mixed.squeeze(0), label


# Model Definitions moved to models.py


# ============================================================
# Test Dataset Classes
# ============================================================

class TestMashupMelDataset(Dataset):
    """Test dataset returning mel spectrograms (for CNN/CRNN)."""
    def __init__(self, mashups_dir, test_csv_path, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration
        self.test_df = pd.read_csv(test_csv_path)
        self.available_files = build_file_lookup(mashups_dir)
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64)
        self.amplitude_to_db = T.AmplitudeToDB()
        print(f"TestMashupMelDataset: {len(self.test_df)} test samples")

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        row = self.test_df.iloc[idx]
        file_id = row['id']
        file_path = find_test_file(file_id, self.available_files, row)
        if file_path is None:
            print(f"WARNING: File not found for id={file_id}")
            return torch.zeros(1, 64, 157), str(file_id)
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - waveform.shape[1]))
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db, str(file_id)


class TestMashupHubertDataset(Dataset):
    """Test dataset returning raw waveforms (for HuBERT)."""
    def __init__(self, mashups_dir, test_csv_path, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration
        self.test_df = pd.read_csv(test_csv_path)
        self.available_files = build_file_lookup(mashups_dir)
        print(f"TestMashupHubertDataset: {len(self.test_df)} test samples")

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        row = self.test_df.iloc[idx]
        file_id = row['id']
        file_path = find_test_file(file_id, self.available_files, row)
        if file_path is None:
            print(f"WARNING: File not found for id={file_id}")
            return torch.zeros(self.num_samples), str(file_id)
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - waveform.shape[1]))
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        return waveform.squeeze(0), str(file_id)


# ============================================================
# Training Functions
# ============================================================

def train_xgboost(X_train, y_labels, paths):
    """Train XGBoost classifier (Milestone 2). Returns clf, label_encoder, val metrics."""
    wandb.init(project="22f1001611-t12026", name="xgboost-mfcc-baseline")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )

    print("Training XGBoost...")
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=SEED, use_label_encoder=False, eval_metric='mlogloss'
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, average='macro')

    print(f"\nXGBoost — Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=GENRES))

    wandb.log({"model": "XGBoost", "val_accuracy": val_acc, "val_macro_f1": val_f1})
    wandb.finish()

    return clf, le, val_acc, val_f1


def train_cnn(model, train_loader, val_loader, device, epochs=10):
    """Train CNN model (Milestone 3). Returns best val F1."""
    wandb.init(project="22f1001611-t12026", name="cnn-from-scratch")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"CNN Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"  Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        wandb.log({"epoch": epoch+1, "train_loss": train_loss/len(train_loader),
                    "train_acc": train_acc, "val_acc": val_acc, "val_macro_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_cnn.pth')
            print(f"  -> New best CNN (F1={val_f1:.4f})")

    wandb.finish()
    return best_f1


def train_crnn(model, train_loader, val_loader, device, epochs=10):
    """Train CRNN model (Milestone 4). Returns best val F1."""
    wandb.init(project="22f1001611-t12026", name="crnn-bilstm")

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"CRNN Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"  Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        wandb.log({"epoch": epoch+1, "train_loss": train_loss/len(train_loader),
                    "train_acc": train_acc, "val_acc": val_acc, "val_macro_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_crnn.pth')
            print(f"  -> New best CRNN (F1={val_f1:.4f})")

    wandb.finish()
    return best_f1


def train_hubert(model, train_loader, val_loader, device, total_epochs=35, phase2_start=11):
    """Train HuBERT with phased strategy (Milestone 5). Returns best val F1."""
    wandb.init(project="22f1001611-t12026", name="hubert-cross-song-35ep-10s")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    best_f1 = 0.0

    for epoch in range(1, total_epochs + 1):
        if epoch == phase2_start:
            print("=" * 50)
            print("PHASE 2: Unfreezing feature encoder!")
            print("=" * 50)
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - phase2_start + 1
            )

        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for waveforms, labels in tqdm(train_loader, desc=f"HuBERT Epoch {epoch}/{total_epochs}"):
            waveforms, labels = waveforms.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(waveforms).logits
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(waveforms).logits
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        phase = 1 if epoch < phase2_start else 2
        print(f"  [Phase {phase}] Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        wandb.log({"epoch": epoch, "phase": phase, "train_loss": train_loss/len(train_loader),
                    "train_acc": train_acc, "val_acc": val_acc, "val_macro_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_hubert.pth')
            print(f"  -> New best HuBERT (F1={val_f1:.4f})")

    wandb.finish()
    return best_f1
