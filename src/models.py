"""
models.py
Contains model definitions: GenreCNN, GenreCRNN
"""

import torch.nn as nn
from utils import NUM_CLASSES

class GenreCNN(nn.Module):
    """CNN for genre classification from mel spectrograms (Milestone 3)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super(GenreCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class GenreCRNN(nn.Module):
    """CRNN: CNN + BiLSTM for genre classification (Milestone 4)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super(GenreCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128 * 8, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])
