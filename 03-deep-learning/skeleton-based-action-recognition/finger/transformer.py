import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')


# ----------------------------
# 1. 手腕归一化函数
# ----------------------------
def normalize_by_wrist(skeleton_seq):
    """将骨架序列以手腕（joint 0）为原点归一化"""
    T = skeleton_seq.shape[0]
    joints = skeleton_seq.reshape(T, 21, 3)  # (T, 21, 3)
    wrist = joints[:, :1, :]  # (T, 1, 3)
    normalized = joints - wrist  # 相对坐标
    return normalized.reshape(T, 63)


# ----------------------------
# 轻量数据增强函数
# ----------------------------
def random_time_crop(seq, target_len):
    T = seq.shape[0]
    if T <= target_len:
        return seq
    start = np.random.randint(0, T - target_len)
    return seq[start:start + target_len]


def spatial_jitter(seq, sigma=0.002):  # 减小抖动幅度
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise


# ----------------------------
# 2. FPHA 数据集加载（仅轻量增强）
# ----------------------------
def load_fpha_dataset(seq_file, label_file, max_seq_len=50,
                      seq_key='train_seq', label_key='train_labels',
                      is_train=True):
    seq_data = sio.loadmat(seq_file)[seq_key][0]  # (num_samples,)
    labels = sio.loadmat(label_file)[label_key].flatten()

    X, y = [], []
    for i, sample in enumerate(seq_data):
        T = sample.shape[1]
        skeleton = sample.T.reshape(T, 21 * 3)  # (T, 63)

        if is_train:
            # 仅轻量增强
            skeleton = random_time_crop(skeleton, max_seq_len)
            skeleton = spatial_jitter(skeleton, sigma=0.002)

        # 填充或截断
        T_new = skeleton.shape[0]
        if T_new < max_seq_len:
            pad = np.zeros((max_seq_len - T_new, 63), dtype=np.float32)
            skeleton = np.vstack([skeleton, pad])
        elif T_new > max_seq_len:
            idx = np.linspace(0, T_new - 1, max_seq_len).astype(np.int64)
            skeleton = skeleton[idx]

        # 手腕归一化
        pos = normalize_by_wrist(skeleton)

        # 速度特征
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = vel[1]

        skeleton = np.concatenate([pos, vel], axis=1)  # (T,126)

        X.append(skeleton)
        y.append(labels[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64) - 1  # 标签从0开始
    return X, y


# ----------------------------
# 3. 加载数据
# ----------------------------
try:
    X_train, y_train = load_fpha_dataset(
        './FPHA/train_seq.mat',
        './FPHA/train_lbl.mat',
        max_seq_len=50,
        seq_key='train_seq',
        label_key='train_labels',
        is_train=True
    )

    X_val, y_val = load_fpha_dataset(
        './FPHA/val_seq.mat',
        './FPHA/val_lbl.mat',
        max_seq_len=50,
        seq_key='val_seq',
        label_key='val_labels',
        is_train=False
    )

    print("训练集:", X_train.shape, y_train.shape)
    print("验证集:", X_val.shape, y_val.shape)
    print("标签范围:", y_train.min(), "~", y_train.max())
    print("类别数:", len(np.unique(y_train)))

except FileNotFoundError as e:
    print(f"错误: 找不到文件 {e.filename}")
    exit(1)


# ----------------------------
# 4. 创建 DataLoader
# ----------------------------
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# ----------------------------
# 5. Mixup 函数
# ----------------------------
def mixup_data(x, y, alpha=0.20):
    """返回混合后的样本和标签"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ----------------------------
# 6. GestureTransformer 模型（保持原样）
# ----------------------------
class GestureTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=8, num_layers=4,
                 num_classes=45, seq_len=50, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.attn_pool = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_fc(x) + self.pos_embedding  # (B,T,D)
        x = x.permute(1, 0, 2)  # (T,B,D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B,T,D)

        B, T, D = x.shape
        t = torch.linspace(-1, 1, T, device=x.device)
        time_bias = torch.exp(-(t / 0.6) ** 2).unsqueeze(0)

        attn = self.attn_pool(x).squeeze(-1) + time_bias
        w = torch.softmax(attn, dim=1).unsqueeze(-1)
        x = (x * w).sum(dim=1)
        return self.classifier(x)


# ----------------------------
# 7. 初始化模型、优化器、损失
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

num_classes = len(np.unique(y_train))
seq_len = X_train.shape[1]
feat_dim = X_train.shape[2]

model = GestureTransformer(input_dim=feat_dim,
                           num_classes=num_classes,
                           seq_len=seq_len).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


# ----------------------------
# 8. 训练循环（加入 Mixup）
# ----------------------------
epochs = 100
train_loss_list, val_acc_list = [], []
best_val_acc, best_epoch = 0.0, 0
patience = 15
trigger_times = 0

print("\n开始训练...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # ---------------- Mixup ----------------
        X_batch, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=0.2)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_loss_list.append(avg_loss)

    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            preds = out.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    acc = correct / total
    val_acc_list.append(acc)

    scheduler.step(acc)

    if acc > best_val_acc:
        best_val_acc = acc
        best_epoch = epoch + 1
        trigger_times = 0
        torch.save(model.state_dict(), 'best_gesture_model.pth')
        print(f"✅ 保存最佳模型，准确率: {acc:.4f}")
    else:
        trigger_times += 1

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    if trigger_times >= patience:
        print(f"早停触发，在第 {epoch + 1} 轮停止")
        break

# ----------------------------
# 9. 输出结果
# ----------------------------
print("\n" + "=" * 50)
print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f} (第 {best_epoch} 轮)")
print("=" * 50)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, marker='x', color='r', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_acc_list, marker='o', color='b', label='Val Acc')
plt.axvline(best_epoch - 1, color='green', linestyle='--', label=f'Best @ {best_epoch}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
