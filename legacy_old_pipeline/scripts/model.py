from typing import Literal

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.classifier(h)


class CRNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 10, rnn_hidden: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=False),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(rnn_hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.freq_pool(h).squeeze(2)
        h = h.transpose(1, 2)
        out, _ = self.rnn(h)
        out = out.mean(dim=1)
        return self.classifier(out)


def build_hubert_model(
    n_classes: int = 10,
    pretrained_name: str = "superb/hubert-base-superb-ks",
    freeze_feature_encoder: bool = False,
):
    try:
        from transformers import HubertForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "transformers is required for HuBERT. Install dependencies from requirements.txt"
        ) from exc

    model = HubertForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    if freeze_feature_encoder:
        model.freeze_feature_encoder()
    return model


def build_model(
    model_type: Literal["cnn", "crnn", "hubert"],
    n_classes: int = 10,
    freeze_feature_encoder: bool = False,
):
    if model_type == "cnn":
        return SimpleCNN(n_classes=n_classes)
    if model_type == "crnn":
        return CRNN(n_classes=n_classes)
    if model_type == "hubert":
        return build_hubert_model(
            n_classes=n_classes,
            freeze_feature_encoder=freeze_feature_encoder,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
