import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CTRDataset(Dataset):
    def __init__(self, img_folder, tgt_folder, img_files, tgt_files, seq_len=256):
        self.img_folder = img_folder
        self.tgt_folder = tgt_folder
        self.img_files = img_files
        self.tgt_files = tgt_files
        self.seq_len = seq_len
        self.cache_img = {}
        self.cache_tgt = {}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.img_files[idx])
        tgt_path = os.path.join(self.tgt_folder, self.tgt_files[idx])

        if img_path not in self.cache_img:
            self.cache_img[img_path] = torch.load(img_path, weights_only=False)
        img_sample = self.cache_img[img_path]
        F2 = img_sample["F2"].float()

        if tgt_path not in self.cache_tgt:
            self.cache_tgt[tgt_path] = torch.load(tgt_path, weights_only=False)
        tgt_sample = self.cache_tgt[tgt_path]
        y = tgt_sample["Intensity"].float()

        assert len(F2) == self.seq_len, f"Form factor length {len(F2)} != {self.seq_len}"
        assert len(y) == self.seq_len, f"Bunch length {len(y)} != {self.seq_len}"

        F2_norm = F2 / (F2.max() + 1e-8)
        y_norm = y / (y.max() + 1e-8)

        return F2_norm.unsqueeze(0), y_norm.unsqueeze(0)


class DilatedResBlock(nn.Module):
    def __init__(self, ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class Dilated_CNN_ResNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.in_conv = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)
        self.block1 = DilatedResBlock(base_ch, dilation=1)
        self.block2 = DilatedResBlock(base_ch, dilation=2)
        self.block3 = DilatedResBlock(base_ch, dilation=4)
        self.block4 = DilatedResBlock(base_ch, dilation=8)
        self.block5 = DilatedResBlock(base_ch, dilation=16)
        self.block6 = DilatedResBlock(base_ch, dilation=32)
        self.out_conv = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.out_conv(x)
        return x


class AsinhMSEWithPhysicsLoss(nn.Module):
    def __init__(self, scale=10.0, lambda_nonneg=1e0, lambda_smooth=1e-2):
        super().__init__()
        self.scale = scale
        self.lambda_nonneg = lambda_nonneg
        self.lambda_smooth = lambda_smooth

    def forward(self, pred, target):
        asinh_loss = torch.mean(
            (torch.asinh(pred / self.scale) - torch.asinh(target / self.scale)) ** 2
        ) * (self.scale ** 2)

        nonneg = torch.mean(torch.relu(-pred) ** 2)
        d2 = pred[:-2] - 2 * pred[1:-1] + pred[2:]
        smooth = torch.mean(d2 ** 2)

        total = (
            asinh_loss
            # + self.lambda_nonneg * nonneg
            # + self.lambda_smooth * smooth
        )
        return total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for img, tgt in loader:
        img = img.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for img, tgt in loader:
            img = img.to(device)
            tgt = tgt.to(device)
            pred = model(img)
            loss = criterion(pred, tgt)
            running_loss += loss.item() * img.size(0)
    return running_loss / len(loader.dataset)
