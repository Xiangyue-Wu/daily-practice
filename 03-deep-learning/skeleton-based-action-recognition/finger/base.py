import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# ----------------------------
# 1. 读取 FPHA 数据集
# ----------------------------
def load_fpha_dataset(seq_file, label_file, max_seq_len=50, seq_key='train_seq', label_key='train_labels'):
    import scipy.io as sio
    import numpy as np

    seq_data = sio.loadmat(seq_file)[seq_key][0]  # (num_samples,)
    labels = sio.loadmat(label_file)[label_key].flatten()

    X, y = [], []
    for i, sample in enumerate(seq_data):
        T = sample.shape[1]
        skeleton = sample.T.reshape(T, 21 * 3)  # (T, 63)
        if T < max_seq_len:
            pad = np.zeros((max_seq_len - T, 63))
            skeleton = np.vstack([skeleton, pad])
        elif T > max_seq_len:
            skeleton = skeleton[:max_seq_len]
        X.append(skeleton)
        y.append(labels[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    y = y - 1  # 标签从0开始
    return X, y


# 训练集
X_train, y_train = load_fpha_dataset(
    './FPHA/train_seq.mat',
    './FPHA/train_lbl.mat',
    max_seq_len=50,
    seq_key='train_seq',
    label_key='train_labels'
)

# 验证集
X_val, y_val = load_fpha_dataset(
    './FPHA/val_seq.mat',
    './FPHA/val_lbl.mat',
    max_seq_len=50,
    seq_key='val_seq',
    label_key='val_labels'
)

print("训练集:", X_train.shape, y_train.shape)
print("验证集:", X_val.shape, y_val.shape)
print("标签范围:", y_train.min(), "~", y_train.max())

# ----------------------------
# 2. 数据标准化
# ----------------------------
num_samples, seq_len, feat_dim = X_train.shape

# 对训练集做标准化
X_train_reshaped = X_train.reshape(-1, feat_dim)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(num_samples, seq_len, feat_dim)

# 验证集也用同一个 scaler
X_val_scaled = scaler.transform(X_val.reshape(-1, feat_dim)).reshape(X_val.shape[0], seq_len, feat_dim)

# 转为 PyTorch Tensor
train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# ----------------------------
# 3. 定义 Transformer 模型
# ----------------------------
class GestureTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2, num_classes=45, seq_len=50, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_fc(x) + self.pos_embedding
        x = x.permute(1, 0, 2)  # Transformer 期望 (seq_len, batch, model_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 对时间维度取平均
        out = self.classifier(x)
        return out


device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(np.unique(y_train))
model = GestureTransformer(input_dim=feat_dim, num_classes=num_classes, seq_len=seq_len).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# 训练与验证
# ----------------------------
epochs = 20
train_loss_list = []
val_acc_list = []
best_val_acc = 0.0  # 记录最佳验证准确率
best_epoch = 0  # 记录最佳准确率对应的epoch

for epoch in range(epochs):
    # 训练
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
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

    # 更新最佳准确率
    if acc > best_val_acc:
        best_val_acc = acc
        best_epoch = epoch + 1

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")

# ----------------------------
# 输出最佳准确率
# ----------------------------
print("\n" + "=" * 50)
print(f"训练完成!")
print(f"最佳验证准确率: {best_val_acc:.4f} (在第 {best_epoch} 个epoch达到)")
print(f"最终验证准确率: {val_acc_list[-1]:.4f}")
print("=" * 50 + "\n")

# ----------------------------
# 可视化训练曲线
# ----------------------------
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), train_loss_list, marker='x', color='r', label='Train Loss')
plt.plot(range(1, epochs + 1), val_acc_list, marker='o', color='b', label='Val Acc')

# 标记最佳准确率点
plt.scatter(best_epoch, best_val_acc, color='green', s=100, zorder=5,
            label=f'Best Acc: {best_val_acc:.4f}')
plt.annotate(f'Best: {best_val_acc:.4f}',
             xy=(best_epoch, best_val_acc),
             xytext=(best_epoch + 1, best_val_acc - 0.02),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, color='green')

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss & Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

