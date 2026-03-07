import argparse
import os
import pickle
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataset import AudioConfig, GENRES, MashupMelDataset, MashupWaveDataset, locate_competition_paths
from scripts.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission for Messy Mashup")
    parser.add_argument("--base-dir", type=str, required=True, help="Path to messy_mashup directory")
    parser.add_argument("--model-type", type=str, choices=["xgb", "cnn", "crnn", "hubert"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="submission.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=96)
    return parser.parse_args()


def _resolve_audio_cfg(args: argparse.Namespace, payload) -> AudioConfig:
    cfg_from_ckpt = payload.get("audio_cfg") if isinstance(payload, dict) else None
    if cfg_from_ckpt:
        return AudioConfig(
            sample_rate=int(cfg_from_ckpt["sample_rate"]),
            duration_sec=float(cfg_from_ckpt["duration_sec"]),
            n_mels=int(cfg_from_ckpt.get("n_mels", args.n_mels)),
            n_fft=int(cfg_from_ckpt.get("n_fft", 1024)),
            hop_length=int(cfg_from_ckpt.get("hop_length", 320)),
        )
    return AudioConfig(sample_rate=args.sample_rate, duration_sec=args.duration, n_mels=args.n_mels)


def _resolve_mashup_file(mashups_dir: str, file_id: str) -> str:
    file_id = str(file_id)
    candidates = [
        os.path.join(mashups_dir, f"{file_id}.wav"),
        os.path.join(mashups_dir, file_id),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Unable to locate mashup file for id={file_id}")


def _load_wave(file_path: str, cfg: AudioConfig) -> np.ndarray:
    waveform, sr = torchaudio.load(file_path, frame_offset=0, num_frames=cfg.num_samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, cfg.sample_rate)
    if waveform.shape[1] > cfg.num_samples:
        waveform = waveform[:, : cfg.num_samples]
    elif waveform.shape[1] < cfg.num_samples:
        waveform = torch.nn.functional.pad(waveform, (0, cfg.num_samples - waveform.shape[1]))
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.squeeze(0).cpu().numpy()


def _extract_features_from_wave(y: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=cfg.sample_rate, n_mfcc=20)
    mel = librosa.feature.melspectrogram(y=y, sr=cfg.sample_rate, n_mels=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=cfg.sample_rate)
    return np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            mel.mean(axis=1),
            mel.std(axis=1),
            chroma.mean(axis=1),
            chroma.std(axis=1),
        ]
    ).astype(np.float32)


def run_xgb_inference(args: argparse.Namespace, paths, payload) -> pd.DataFrame:
    model = payload["model"]
    genres = payload.get("genres", GENRES)
    cfg = _resolve_audio_cfg(args, payload)

    test_df = pd.read_csv(paths["test_csv"], dtype={"id": str})
    feats: List[np.ndarray] = []
    file_ids: List[str] = []

    for file_id in tqdm(test_df["id"].tolist(), desc="Extracting test features"):
        fp = _resolve_mashup_file(paths["mashups_dir"], file_id)
        wave = _load_wave(fp, cfg)
        feats.append(_extract_features_from_wave(wave, cfg))
        file_ids.append(file_id)

    X = np.stack(feats)
    pred_idx = model.predict(X)
    pred_labels = [genres[int(i)] for i in pred_idx]
    return pd.DataFrame({"id": file_ids, "genre": pred_labels})


def run_neural_inference(args: argparse.Namespace, paths, payload) -> pd.DataFrame:
    model_type = payload.get("model_type", args.model_type)
    if model_type != args.model_type:
        raise ValueError(f"Checkpoint model_type={model_type} but arg model_type={args.model_type}")

    genres = payload.get("genres", GENRES)
    cfg = _resolve_audio_cfg(args, payload)

    if model_type == "hubert":
        dataset = MashupWaveDataset(paths["mashups_dir"], paths["test_csv"], cfg)
    else:
        dataset = MashupMelDataset(paths["mashups_dir"], paths["test_csv"], cfg)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    if model_type == "hubert" and payload.get("model_config"):
        from transformers import HubertConfig, HubertForSequenceClassification

        config = HubertConfig.from_dict(payload["model_config"])
        model = HubertForSequenceClassification(config)
    else:
        model = build_model(
            model_type=model_type,
            n_classes=len(genres),
            freeze_feature_encoder=payload.get("freeze_feature_encoder", False),
        )
    model.load_state_dict(payload["state_dict"], strict=True)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    all_ids: List[str] = []
    all_preds: List[str] = []

    with torch.no_grad():
        for inputs, file_ids in tqdm(loader, desc="Predicting"):
            inputs = inputs.to(device, non_blocking=True)
            if model_type == "hubert":
                logits = model(inputs).logits
            else:
                logits = model(inputs)
            pred_idx = torch.argmax(logits, dim=1).cpu().tolist()
            for fid, idx in zip(file_ids, pred_idx):
                all_ids.append(str(fid))
                all_preds.append(genres[int(idx)])

    return pd.DataFrame({"id": all_ids, "genre": all_preds})


def main() -> None:
    args = parse_args()
    paths = locate_competition_paths(args.base_dir)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.model_type == "xgb":
        with open(checkpoint_path, "rb") as f:
            payload = pickle.load(f)
        submission_df = run_xgb_inference(args, paths, payload)
    else:
        payload = torch.load(checkpoint_path, map_location="cpu")
        submission_df = run_neural_inference(args, paths, payload)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_path, index=False)
    print(f"Saved submission to: {out_path}")


if __name__ == "__main__":
    main()
