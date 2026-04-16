"""
inference.py — Test inference, TTA, and ensemble submission generation.
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from utils import (
    SAMPLE_RATE, DURATION, NUM_CLASSES, IDX_TO_GENRE,
    TTA_WINDOW_STARTS_SEC, TTA_CROPS,
    W_CNN, W_CRNN, W_HUBERT,
    get_paths, build_file_lookup, find_test_file,
    load_systematic_tta_crops, extract_features_from_file,
)
from models import GenreCNN, GenreCRNN


def run_hubert_inference(model, paths, device):
    """Run HuBERT inference with systematic TTA. Returns probs array and IDs."""
    model.eval()
    test_df = pd.read_csv(paths['TEST_CSV'])
    available_files = build_file_lookup(paths['MASHUPS_DIR'])

    all_probs, all_ids = [], []

    print(f"Running HuBERT inference with systematic TTA ({TTA_CROPS} windows)...")
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="HuBERT TTA"):
            file_id = row['id']
            file_path = find_test_file(file_id, available_files, row)

            if file_path is None:
                print(f"WARNING: File not found for id={file_id}")
                all_probs.append(np.ones(NUM_CLASSES) / NUM_CLASSES)
                all_ids.append(str(file_id))
                continue

            crops = load_systematic_tta_crops(file_path).to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(crops).logits
            probs = F.softmax(outputs, dim=1).mean(dim=0).cpu().numpy()
            all_probs.append(probs)
            all_ids.append(str(file_id))

    all_probs = np.array(all_probs)
    np.save('hubert_tta_probs.npy', all_probs)

    predictions = np.argmax(all_probs, axis=1)
    pred_genres = [IDX_TO_GENRE[p] for p in predictions]

    test_df_orig = pd.read_csv(paths['TEST_CSV'])
    sub_ids = [int(x) for x in all_ids] if test_df_orig['id'].dtype == int else all_ids
    submission_df = pd.DataFrame({'id': sub_ids, 'genre': pred_genres})
    submission_df.to_csv('submission.csv', index=False)
    submission_df.to_csv('submission_hubert.csv', index=False)

    print(f"HuBERT submission saved: submission.csv")
    print(submission_df['genre'].value_counts())
    return all_probs, all_ids


def run_ensemble_inference(cnn_model, crnn_model, hubert_model, paths, device):
    """Run weighted ensemble (CNN + CRNN + HuBERT) with systematic TTA."""
    cnn_model.eval()
    crnn_model.eval()
    hubert_model.eval()

    mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    amplitude_to_db = T.AmplitudeToDB()

    hubert_tta_probs = np.load('hubert_tta_probs.npy')
    test_df = pd.read_csv(paths['TEST_CSV'])
    available_files = build_file_lookup(paths['MASHUPS_DIR'])
    num_samples = SAMPLE_RATE * DURATION

    ensemble_preds, ensemble_ids = [], []

    print(f"Running ensemble inference (CNN={W_CNN}, CRNN={W_CRNN}, HuBERT={W_HUBERT})...")
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Ensemble TTA"):
            file_id = row['id']
            file_path = find_test_file(file_id, available_files, row)

            if file_path is None:
                pred = np.argmax(hubert_tta_probs[idx])
                ensemble_preds.append(IDX_TO_GENRE[pred])
                ensemble_ids.append(file_id)
                continue

            waveform, sr = torchaudio.load(file_path)
            if sr != SAMPLE_RATE:
                waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            total_len = waveform.shape[1]

            cnn_probs_list, crnn_probs_list = [], []
            for start_sec in TTA_WINDOW_STARTS_SEC:
                start_sample = int(start_sec * SAMPLE_RATE)
                end_sample = start_sample + num_samples

                if end_sample <= total_len:
                    crop = waveform[:, start_sample:end_sample]
                elif start_sample < total_len:
                    crop = waveform[:, start_sample:]
                    crop = torch.nn.functional.pad(crop, (0, num_samples - crop.shape[1]))
                else:
                    if total_len >= num_samples:
                        crop = waveform[:, total_len - num_samples:]
                    else:
                        crop = torch.nn.functional.pad(waveform, (0, num_samples - total_len))

                crop = crop / (torch.max(torch.abs(crop)) + 1e-8)
                mel_spec = mel_transform(crop)
                mel_spec_db = amplitude_to_db(mel_spec)
                mel_input = mel_spec_db.unsqueeze(0).to(device)

                with torch.amp.autocast('cuda'):
                    cnn_logits = cnn_model(mel_input)
                    crnn_logits = crnn_model(mel_input)

                cnn_probs_list.append(F.softmax(cnn_logits[0], dim=0).cpu().numpy())
                crnn_probs_list.append(F.softmax(crnn_logits[0], dim=0).cpu().numpy())

            cnn_probs = np.mean(cnn_probs_list, axis=0)
            crnn_probs = np.mean(crnn_probs_list, axis=0)
            hub_probs = hubert_tta_probs[idx]

            combined = W_CNN * cnn_probs + W_CRNN * crnn_probs + W_HUBERT * hub_probs
            pred = np.argmax(combined)
            ensemble_preds.append(IDX_TO_GENRE[pred])
            ensemble_ids.append(file_id)

    sub_ensemble = pd.DataFrame({'id': ensemble_ids, 'genre': ensemble_preds})
    if test_df['id'].dtype != sub_ensemble['id'].dtype:
        sub_ensemble['id'] = sub_ensemble['id'].astype(test_df['id'].dtype)
    sub_ensemble.to_csv('submission_ensemble.csv', index=False)

    print(f"Ensemble submission saved: submission_ensemble.csv")
    print(sub_ensemble['genre'].value_counts())
    return sub_ensemble


def run_xgboost_inference(clf, label_encoder, paths):
    """Run XGBoost inference on test set."""
    test_df = pd.read_csv(paths['TEST_CSV'])
    available_files = build_file_lookup(paths['MASHUPS_DIR'])

    X_test, test_ids = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="XGBoost test features"):
        file_id = str(row['id'])
        file_path = find_test_file(file_id, available_files, row)
        if file_path is None:
            print(f"WARNING: File not found for id={file_id}, using zeros")
            X_test.append(np.zeros(64))
        else:
            X_test.append(extract_features_from_file(file_path))
        test_ids.append(int(file_id))

    X_test = np.array(X_test)
    y_pred = clf.predict(X_test)
    y_genres = label_encoder.inverse_transform(y_pred)

    sub = pd.DataFrame({'id': test_ids, 'genre': y_genres})
    sub.to_csv('submission_xgb.csv', index=False)
    print(f"XGBoost submission saved: submission_xgb.csv")
    return sub
