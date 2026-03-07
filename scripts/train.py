import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.dataset import MashupDataset, list_audio_files
from scripts.model import SimpleCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='data/raw/mashups')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        # placeholder: no real labels here
        wave = batch['wave']
        wave = torch.from_numpy(wave).unsqueeze(1).to(device)
        # create dummy target
        target = torch.zeros(wave.size(0), dtype=torch.long).to(device)
        out = model(wave)
        loss = loss_fn(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()


def main():
    args = parse_args()
    files = list_audio_files(args.data_dir)
    ds = MashupDataset(files[:32], duration=5.0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = args.device
    model = SimpleCNN(in_channels=1, n_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        train_one_epoch(model, loader, opt, device)
        print(f'Epoch {ep+1} done')
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/baseline.pth')


if __name__ == '__main__':
    main()
