import os
from typing import List
import numpy as np
import soundfile as sf
import librosa
from torch.utils.data import Dataset


class MashupDataset(Dataset):
    """Simple dataset stub that loads a list of file paths and returns waveform + id."""

    def __init__(self, file_paths: List[str], sr: int = 22050, duration: float = 30.0, transform=None):
        self.file_paths = file_paths
        self.sr = sr
        self.duration = duration
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        wav, _ = librosa.load(fp, sr=self.sr, mono=True)
        # pad or trim to fixed length
        target_len = int(self.sr * self.duration)
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]
        if self.transform:
            wav = self.transform(wav)
        return {
            "id": os.path.basename(fp).split('.')[0],
            "wave": wav.astype(np.float32)
        }


def list_audio_files(root_dir: str, exts=('.wav', '.flac')):
    files = []
    for r, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.join(r, f))
    return sorted(files)


if __name__ == '__main__':
    # quick local test
    from pprint import pprint
    files = list_audio_files('data/raw/mashups')
    print('Found', len(files), 'mashups')
    if files:
        ds = MashupDataset(files[:2], duration=10.0)
        for item in ds:
            pprint({'id': item['id'], 'wave_len': len(item['wave'])})
