import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (B, C, F, T)
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class CRNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10, rnn_hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.rnn = nn.GRU(16, rnn_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden * 2, n_classes)

    def forward(self, x):
        # x: (B, C, F, T)
        h = self.conv(x)  # (B, C2, F2, T2)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        out, _ = self.rnn(h)
        out = out.mean(1)
        return self.fc(out)


if __name__ == '__main__':
    m = SimpleCNN()
    x = torch.randn(2, 1, 128, 128)
    print(m(x).shape)
