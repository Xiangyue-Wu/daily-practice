# -*- coding: utf-8 -*-
"""
FPHA Gesture Recognition with Single-Layer LSTM (Fixed Version)
✅ Added: Root-relative normalization + per-sample standardization
"""

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import random

# ======================
# 配置参数
# ======================
DATA_DIR = "../FPHA"
MAX_LEN = 50
NUM_CLASSES = 45
INPUT_SIZE = 63
HIDDEN_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0005  # 降低学习率
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)


# ======================
# 数据预处理：归一化 + 填充
# ======================
def normalize_skeleton(seq):
    """
    对 (T, 63) 序列进行归一化：
    1. 转为 (T, 21, 3)
    2. 以手腕（joint 0）为原点
    3. 对非零帧做标准化（避免 padding 干扰）
    """
    T_orig = seq.shape[0]
    joints = seq.reshape(T_orig, 21, 3)  # (T, 21, 3)

    # Step 1: 以手腕为原点
    wrist = joints[:, 0:1, :]  # (T, 1, 3)
    joints_centered = joints - wrist  # (T, 21, 3)

    # Step 2: 仅对有效帧（非全零）计算统计量
    # 注意：padding 部分是全零，我们跳过它们
    valid_mask = ~(np.all(joints_centered == 0, axis=(1, 2)))  # (T,)
    if np.any(valid_mask):
        valid_joints = joints_centered[valid_mask]  # (T_valid, 21, 3)
        mean = valid_joints.mean(axis=(0, 1), keepdims=True)  # (1, 1, 3)
        std = valid_joints.std(axis=(0, 1), keepdims=True) + 1e-8  # (1, 1, 3)
        joints_norm = (joints_centered - mean) / std
    else:
        joints_norm = joints_centered  # fallback

    return joints_norm.reshape(T_orig, 63)


def pad_sequence(seq, max_len=MAX_LEN):
    T = seq.shape[0]
    if T >= max_len:
        return seq[:max_len]
    else:
        pad = np.zeros((max_len - T, INPUT_SIZE), dtype=np.float32)
        return np.vstack([seq, pad])


# ======================
# Dataset（含归一化）
# ======================
class FPHADataset(Dataset):
    def __init__(self, seq_path, lbl_path, is_train=True):
        seq_key = 'train_seq' if is_train else 'val_seq'
        lbl_key = 'train_labels' if is_train else 'val_labels'

        seq_mat = sio.loadmat(seq_path)[seq_key]
        lbl_mat = sio.loadmat(lbl_path)[lbl_key].flatten()

        self.sequences = []
        self.labels = []

        for i in tqdm(range(seq_mat.shape[1]), desc=f"Loading {'train' if is_train else 'val'}"):
            raw_seq = seq_mat[0, i].T  # (T, 63)
            padded = pad_sequence(raw_seq, MAX_LEN)  # (100, 63)
            normalized = normalize_skeleton(padded)  # ✅ 关键修复！
            self.sequences.append(normalized.astype(np.float32))
            self.labels.append(lbl_mat[i] - 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


# ======================
# LSTM 模型
# ======================
class LSTMGestureClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=45):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))


# ======================
# 训练函数
# ======================
def train_one_run(seed):
    set_seed(seed)

    # 加载数据
    train_dataset = FPHADataset(os.path.join(DATA_DIR, "train_seq.mat"),
                                os.path.join(DATA_DIR, "train_lbl.mat"), is_train=True)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataset = FPHADataset(os.path.join(DATA_DIR, "val_seq.mat"),
                              os.path.join(DATA_DIR, "val_lbl.mat"), is_train=False)
    val_X = torch.stack([x for x, _ in val_dataset])
    val_y = torch.tensor([y for _, y in val_dataset], dtype=torch.long)
    val_X, val_y = val_X.to(DEVICE), val_y.to(DEVICE)

    # 模型 & 优化器
    model = LSTMGestureClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        with torch.no_grad():
            val_out = model(val_X)
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred == val_y).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Seed {seed} | Epoch {epoch + 1:2d} | "
                  f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return best_val_acc


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("Starting FIXED LSTM baseline experiments...\n")

    results = []
    num_runs = 3  # 可先跑3次快速验证
    for run in range(num_runs):
        print(f"\n{'=' * 50}")
        print(f"Run {run + 1}/{num_runs}")
        print(f"{'=' * 50}")
        acc = train_one_run(seed=run)
        results.append(acc)
        print(f"✅ Best validation accuracy: {acc:.4f}")

    mean_acc = np.mean(results)
    std_acc = np.std(results)

    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS (FIXED LSTM Baseline)")
    print("=" * 50)
    print(f"Runs: {[f'{r:.4f}' for r in results]}")
    print(f"Mean: {mean_acc:.4f}")
    print(f"Std:  {std_acc:.4f}")