import torch
import torch.nn.functional as F
import numpy as np

def systematic_tta(model, audio_tensor, duration=30, window_len=10, overlap=5):
    """
    Test Time Augmentation (TTA)
    Extracts 5 overlapping 10-second windows from a 30s audio file:
    [0-10s], [5-15s], [10-20s], [15-25s], [20-30s]
    Computes predictions for each and averages them.
    """
    pass # Implementation from the notebook

def ensemble_predict(cnn_probs, crnn_probs, hubert_probs, weights={'cnn': 0.15, 'crnn': 0.20, 'hubert': 0.65}):
    """
    Weighted Soft Voting Ensemble to combine the output probabilities
    of all 3 trained models.
    """
    combined_probs = (weights['cnn'] * cnn_probs) + \
                     (weights['crnn'] * crnn_probs) + \
                     (weights['hubert'] * hubert_probs)
    
    final_prediction = torch.argmax(combined_probs, dim=1)
    return final_prediction
