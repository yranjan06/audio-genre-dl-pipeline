import argparse
import contextlib
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataset import (
    AudioConfig,
    GENRES,
    STEM_FILES,
    StemMelDataset,
    StemWaveDataset,
    build_train_val_indices,
    list_noise_files,
    list_song_dirs,
    locate_competition_paths,
)
from scripts.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models for Messy Mashup")
    parser.add_argument("--base-dir", type=str, required=True, help="Path to messy_mashup directory")
    parser.add_argument("--model-type", type=str, choices=["xgb", "cnn", "crnn", "hubert"], required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=96)
    parser.add_argument("--noise-prob", type=float, default=0.7)
    parser.add_argument("--max-songs-per-genre", type=int, default=None)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-feature-encoder", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dl-genai-messy-mashup")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_init_wandb(args: argparse.Namespace, config: Dict):
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Continuing without wandb logging.")
        return None

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("WANDB_API_KEY not found. Continuing without wandb logging.")
        return None

    wandb.login(key=api_key)
    run_name = args.run_name or f"{args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run = wandb.init(project=args.wandb_project, name=run_name, config=config)
    return run


def ensure_paths_exist(paths: Dict[str, str]) -> None:
    required = ["stems_dir", "mashups_dir", "test_csv"]
    for key in required:
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"Required path missing: {key} -> {paths[key]}")


def _load_mixed_song_wave(song_dir: str, cfg: AudioConfig) -> np.ndarray:
    mixed = torch.zeros((1, cfg.num_samples), dtype=torch.float32)
    for stem_name in STEM_FILES:
        stem_path = os.path.join(song_dir, stem_name)
        if not os.path.exists(stem_path):
            continue
        waveform, sr = torchaudio.load(stem_path, frame_offset=0, num_frames=cfg.num_samples)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, cfg.sample_rate)
        if waveform.shape[1] > cfg.num_samples:
            waveform = waveform[:, : cfg.num_samples]
        elif waveform.shape[1] < cfg.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, cfg.num_samples - waveform.shape[1]))
        mixed += waveform
    mixed = mixed / (mixed.abs().max() + 1e-8)
    return mixed.squeeze(0).cpu().numpy()


def _extract_song_features(song_dir: str, cfg: AudioConfig) -> np.ndarray:
    import librosa

    y = _load_mixed_song_wave(song_dir, cfg)
    mfcc = librosa.feature.mfcc(y=y, sr=cfg.sample_rate, n_mfcc=20)
    mel = librosa.feature.melspectrogram(y=y, sr=cfg.sample_rate, n_mels=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=cfg.sample_rate)

    feature_vector = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            mel.mean(axis=1),
            mel.std(axis=1),
            chroma.mean(axis=1),
            chroma.std(axis=1),
        ]
    )
    return feature_vector.astype(np.float32)


def train_xgb(
    args: argparse.Namespace,
    cfg: AudioConfig,
    song_items: List[Tuple[str, int]],
    train_indices: List[int],
    val_indices: List[int],
    output_dir: Path,
    wandb_run,
) -> Dict[str, float]:
    try:
        import xgboost as xgb

        model_cls = "xgboost"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        model_cls = "random_forest"

    print("Extracting classical features...")
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for idx in tqdm(train_indices, desc="Train feature extraction"):
        song_dir, label = song_items[idx]
        X_train.append(_extract_song_features(song_dir, cfg))
        y_train.append(label)

    for idx in tqdm(val_indices, desc="Val feature extraction"):
        song_dir, label = song_items[idx]
        X_val.append(_extract_song_features(song_dir, cfg))
        y_val.append(label)

    X_train_np = np.stack(X_train)
    y_train_np = np.asarray(y_train)
    X_val_np = np.stack(X_val)
    y_val_np = np.asarray(y_val)

    if model_cls == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softmax",
            num_class=len(GENRES),
            eval_metric="mlogloss",
            random_state=args.seed,
            n_jobs=max(1, args.num_workers),
        )
    else:
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=args.seed,
            n_jobs=max(1, args.num_workers),
            class_weight="balanced",
        )

    model.fit(X_train_np, y_train_np)
    val_pred = model.predict(X_val_np)

    metrics = {
        "val_accuracy": float(accuracy_score(y_val_np, val_pred)),
        "val_macro_f1": float(f1_score(y_val_np, val_pred, average="macro")),
    }

    payload = {
        "model": model,
        "audio_cfg": cfg.__dict__,
        "genres": GENRES,
        "model_backend": model_cls,
    }
    model_path = output_dir / "best_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    if wandb_run is not None:
        import wandb

        wandb.log(metrics)

    print(f"Saved classical model to: {model_path}")
    return metrics


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    model_type: str,
    train: bool,
    amp_enabled: bool,
    scaler,
) -> Tuple[float, float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            amp_context = (
                torch.cuda.amp.autocast(enabled=amp_enabled)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp_context:
                if model_type == "hubert":
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            all_targets.extend(labels.detach().cpu().tolist())
            all_preds.extend(pred.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    return avg_loss, acc, macro_f1


def train_neural(
    args: argparse.Namespace,
    cfg: AudioConfig,
    paths: Dict[str, str],
    song_items: List[Tuple[str, int]],
    train_indices: List[int],
    val_indices: List[int],
    output_dir: Path,
    wandb_run,
) -> Dict[str, float]:
    train_items = [song_items[i] for i in train_indices]
    val_items = [song_items[i] for i in val_indices]
    noise_files = list_noise_files(paths["noise_dir"])

    if args.model_type == "hubert":
        train_ds = StemWaveDataset(
            song_items=train_items,
            noise_files=noise_files,
            audio_cfg=cfg,
            noise_prob=args.noise_prob,
            random_offset=True,
        )
        val_ds = StemWaveDataset(
            song_items=val_items,
            noise_files=noise_files,
            audio_cfg=cfg,
            noise_prob=args.noise_prob,
            random_offset=False,
        )
    else:
        train_ds = StemMelDataset(
            song_items=train_items,
            noise_files=noise_files,
            audio_cfg=cfg,
            noise_prob=args.noise_prob,
            random_offset=True,
        )
        val_ds = StemMelDataset(
            song_items=val_items,
            noise_files=noise_files,
            audio_cfg=cfg,
            noise_prob=args.noise_prob,
            random_offset=False,
        )

    pin_memory = args.device.startswith("cuda") and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    model = build_model(
        model_type=args.model_type,
        n_classes=len(GENRES),
        freeze_feature_encoder=args.freeze_feature_encoder,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss()

    amp_enabled = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    ckpt_path = output_dir / f"best_{args.model_type}.pth"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_type=args.model_type,
            train=True,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )

        val_loss, val_acc, val_f1 = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_type=args.model_type,
            train=False,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_accuracy": tr_acc,
            "train_macro_f1": tr_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        if wandb_run is not None:
            import wandb

            wandb.log(epoch_metrics)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = {
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "train_accuracy": tr_acc,
                "train_macro_f1": tr_f1,
            }
            payload = {
                "model_type": args.model_type,
                "state_dict": model.state_dict(),
                "genres": GENRES,
                "audio_cfg": cfg.__dict__,
                "freeze_feature_encoder": args.freeze_feature_encoder,
            }
            if args.model_type == "hubert":
                payload["model_config"] = model.config.to_dict()
            torch.save(payload, ckpt_path)
            print(f"Saved new best checkpoint: {ckpt_path}")

    return best_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    paths = locate_competition_paths(args.base_dir)
    ensure_paths_exist(paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = AudioConfig(
        sample_rate=args.sample_rate,
        duration_sec=args.duration,
        n_mels=args.n_mels,
    )

    song_items = list_song_dirs(
        paths["stems_dir"],
        genres=GENRES,
        max_songs_per_genre=args.max_songs_per_genre,
    )
    if len(song_items) < 20:
        raise RuntimeError("Too few songs discovered. Please verify base-dir and dataset structure.")

    train_indices, val_indices = build_train_val_indices(
        song_items=song_items,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    config = {
        "model_type": args.model_type,
        "num_songs": len(song_items),
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "audio_cfg": cfg.__dict__,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "noise_prob": args.noise_prob,
        "freeze_feature_encoder": args.freeze_feature_encoder,
    }

    wandb_run = maybe_init_wandb(args, config)
    try:
        if args.model_type == "xgb":
            metrics = train_xgb(
                args=args,
                cfg=cfg,
                song_items=song_items,
                train_indices=train_indices,
                val_indices=val_indices,
                output_dir=output_dir,
                wandb_run=wandb_run,
            )
        else:
            metrics = train_neural(
                args=args,
                cfg=cfg,
                paths=paths,
                song_items=song_items,
                train_indices=train_indices,
                val_indices=val_indices,
                output_dir=output_dir,
                wandb_run=wandb_run,
            )

        metrics_path = output_dir / f"metrics_{args.model_type}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Final metrics saved to: {metrics_path}")
        print(json.dumps(metrics, indent=2))
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
