import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

GENRES: List[str] = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

STEM_FILES: Tuple[str, ...] = ("vocals.wav", "drums.wav", "bass.wav", "others.wav")


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration_sec: float = 5.0
    n_mels: int = 96
    n_fft: int = 1024
    hop_length: int = 320

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_sec)


def detect_noise_dir(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "ESC-50-master", "audio"),
        os.path.join(base_dir, "ESC-50", "audio"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def list_noise_files(noise_dir: str) -> List[str]:
    if not os.path.isdir(noise_dir):
        return []
    return sorted(
        str(Path(noise_dir) / name)
        for name in os.listdir(noise_dir)
        if name.lower().endswith(".wav")
    )


def list_song_dirs(
    stems_dir: str,
    genres: Sequence[str] = GENRES,
    max_songs_per_genre: Optional[int] = None,
) -> List[Tuple[str, int]]:
    song_items: List[Tuple[str, int]] = []
    for label_idx, genre in enumerate(genres):
        genre_dir = Path(stems_dir) / genre
        if not genre_dir.exists():
            continue
        song_dirs = [p for p in genre_dir.iterdir() if p.is_dir()]
        song_dirs = sorted(song_dirs)
        if max_songs_per_genre is not None:
            song_dirs = song_dirs[: max(0, max_songs_per_genre)]
        for song_dir in song_dirs:
            song_items.append((str(song_dir), label_idx))
    return song_items


def build_train_val_indices(
    song_items: Sequence[Tuple[str, int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    by_label: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(song_items):
        by_label.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for label_indices in by_label.values():
        copied = label_indices[:]
        rng.shuffle(copied)
        val_count = max(1, int(len(copied) * val_ratio))
        val_indices.extend(copied[:val_count])
        train_indices.extend(copied[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _pad_or_trim(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    if waveform.shape[1] > target_len:
        return waveform[:, :target_len]
    if waveform.shape[1] < target_len:
        return torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
    return waveform


def _load_wave_segment(
    file_path: str,
    sample_rate: int,
    num_samples: int,
    random_offset: bool,
) -> torch.Tensor:
    frame_offset = 0
    if random_offset:
        info = torchaudio.info(file_path)
        if info.num_frames > num_samples:
            frame_offset = random.randint(0, info.num_frames - num_samples)
    waveform, sr = torchaudio.load(
        file_path,
        frame_offset=frame_offset,
        num_frames=num_samples,
    )
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return _pad_or_trim(waveform, num_samples)


def _normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    return waveform / (waveform.abs().max() + 1e-8)


def _resolve_mashup_path(mashups_dir: str, file_id: str) -> str:
    file_id = str(file_id)
    candidates = [
        os.path.join(mashups_dir, file_id),
        os.path.join(mashups_dir, f"{file_id}.wav"),
    ]
    if not file_id.lower().endswith(".wav"):
        candidates.reverse()
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve mashup file for id={file_id}")


class _StemMixBase(Dataset):
    def __init__(
        self,
        song_items: Sequence[Tuple[str, int]],
        noise_files: Optional[Sequence[str]],
        audio_cfg: AudioConfig,
        noise_prob: float = 0.7,
        stem_gain_range: Tuple[float, float] = (0.8, 1.2),
        noise_gain_range: Tuple[float, float] = (0.08, 0.35),
        random_offset: bool = True,
    ):
        self.song_items = list(song_items)
        self.noise_files = list(noise_files or [])
        self.audio_cfg = audio_cfg
        self.noise_prob = noise_prob
        self.stem_gain_range = stem_gain_range
        self.noise_gain_range = noise_gain_range
        self.random_offset = random_offset

    def __len__(self) -> int:
        return len(self.song_items)

    def _mix_song(self, idx: int) -> Tuple[torch.Tensor, int]:
        song_dir, label = self.song_items[idx]
        mixed = torch.zeros((1, self.audio_cfg.num_samples), dtype=torch.float32)

        for stem_name in STEM_FILES:
            stem_path = os.path.join(song_dir, stem_name)
            if not os.path.exists(stem_path):
                continue
            stem_wave = _load_wave_segment(
                stem_path,
                sample_rate=self.audio_cfg.sample_rate,
                num_samples=self.audio_cfg.num_samples,
                random_offset=self.random_offset,
            )
            gain = random.uniform(*self.stem_gain_range)
            mixed += stem_wave * gain

        if self.noise_files and random.random() <= self.noise_prob:
            noise_path = random.choice(self.noise_files)
            noise_wave = _load_wave_segment(
                noise_path,
                sample_rate=self.audio_cfg.sample_rate,
                num_samples=self.audio_cfg.num_samples,
                random_offset=True,
            )
            shift = random.randint(0, self.audio_cfg.num_samples - 1)
            noise_wave = torch.roll(noise_wave, shifts=shift, dims=1)
            noise_gain = random.uniform(*self.noise_gain_range)
            mixed += noise_wave * noise_gain

        return _normalize_audio(mixed), label


class StemWaveDataset(_StemMixBase):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        waveform, label = self._mix_song(idx)
        return waveform.squeeze(0), label


class StemMelDataset(_StemMixBase):
    def __init__(
        self,
        song_items: Sequence[Tuple[str, int]],
        noise_files: Optional[Sequence[str]],
        audio_cfg: AudioConfig,
        noise_prob: float = 0.7,
        stem_gain_range: Tuple[float, float] = (0.8, 1.2),
        noise_gain_range: Tuple[float, float] = (0.08, 0.35),
        random_offset: bool = True,
    ):
        super().__init__(
            song_items=song_items,
            noise_files=noise_files,
            audio_cfg=audio_cfg,
            noise_prob=noise_prob,
            stem_gain_range=stem_gain_range,
            noise_gain_range=noise_gain_range,
            random_offset=random_offset,
        )
        self.mel_transform = T.MelSpectrogram(
            sample_rate=audio_cfg.sample_rate,
            n_fft=audio_cfg.n_fft,
            hop_length=audio_cfg.hop_length,
            n_mels=audio_cfg.n_mels,
        )
        self.db_transform = T.AmplitudeToDB(stype="power")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        waveform, label = self._mix_song(idx)
        mel = self.db_transform(self.mel_transform(waveform))
        return mel, label


class MashupWaveDataset(Dataset):
    def __init__(self, mashups_dir: str, test_csv_path: str, audio_cfg: AudioConfig):
        self.mashups_dir = mashups_dir
        self.audio_cfg = audio_cfg
        self.test_df = pd.read_csv(test_csv_path, dtype={"id": str})

    def __len__(self) -> int:
        return len(self.test_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        file_id = str(self.test_df.iloc[idx]["id"])
        file_path = _resolve_mashup_path(self.mashups_dir, file_id)
        waveform = _load_wave_segment(
            file_path,
            sample_rate=self.audio_cfg.sample_rate,
            num_samples=self.audio_cfg.num_samples,
            random_offset=False,
        )
        waveform = _normalize_audio(waveform)
        return waveform.squeeze(0), file_id


class MashupMelDataset(Dataset):
    def __init__(self, mashups_dir: str, test_csv_path: str, audio_cfg: AudioConfig):
        self.wave_ds = MashupWaveDataset(mashups_dir, test_csv_path, audio_cfg)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=audio_cfg.sample_rate,
            n_fft=audio_cfg.n_fft,
            hop_length=audio_cfg.hop_length,
            n_mels=audio_cfg.n_mels,
        )
        self.db_transform = T.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.wave_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        waveform, file_id = self.wave_ds[idx]
        mel = self.db_transform(self.mel_transform(waveform.unsqueeze(0)))
        return mel, file_id


def locate_competition_paths(base_dir: str) -> Dict[str, str]:
    stems_dir = os.path.join(base_dir, "genres_stems")
    mashups_dir = os.path.join(base_dir, "mashups")
    test_csv = os.path.join(base_dir, "test.csv")
    noise_dir = detect_noise_dir(base_dir)
    return {
        "base_dir": base_dir,
        "stems_dir": stems_dir,
        "mashups_dir": mashups_dir,
        "test_csv": test_csv,
        "noise_dir": noise_dir,
    }
