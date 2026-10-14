import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)

# ==================== MSTAR数据集类 ====================
class MSTARDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(128, 128)):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        self.images = []
        self.labels = []

        self.classes = sorted(
            [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        print(f"发现类别: {self.classes}")
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                for img_path in class_dir.glob(ext):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        print(f"加载了 {len(self.images)} 张图片，来自 {self.num_classes} 个类别")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        if self.transform:
            image = self.transform(image)
        return image, label


# 分层抽样
def get_stratified_subset(full_dataset, ratio, seed):
    random.seed(seed)
    np.random.seed(seed)

    targets = np.array(full_dataset.labels)
    classes = np.unique(targets)
    selected_indices = []

    print(f"\n--- 正在执行分层抽样 (比例: {ratio}, 种子: {seed}) ---")
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        num_samples = int(len(cls_indices) * ratio)
        sampled = np.random.choice(cls_indices, num_samples, replace=False)
        selected_indices.extend(sampled.tolist())
        print(f"  类别 '{full_dataset.classes[cls]}': 总数 {len(cls_indices)} -> 抽取 {num_samples}")

    random.shuffle(selected_indices)
    print(f"已抽样完成！共选取 {len(selected_indices)} 张图片用于训练。\n" + "-" * 60)

    return Subset(full_dataset, selected_indices)

# ==================== 训练与评估 ====================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(model(images).detach(), 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        if batch_idx % 5 == 0:
            print(f"  Epoch {epoch + 1} | 批次 [{batch_idx}/{len(loader)}] | Loss: {loss.item():.4f}", flush=True)
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds) * 100, all_preds, all_labels


# ==================== 可视化1：训练曲线 ====================
def plot_training_curves(train_losses, train_accs, test_accs, best_acc, save_path='training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss曲线
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=1.5)
    axes[1].plot(epochs, test_accs, 'r-', label='Test Acc', linewidth=1.5)
    axes[1].axhline(y=best_acc, color='g', linestyle='--', label=f'Best Test Acc = {best_acc:.2f}%')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Train & Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存为 {save_path}")


# ==================== 可视化2：混淆矩阵 ====================
def plot_confusion_matrix(all_labels, all_preds, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=100)

    # 在每个格子标百分比
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f'{cm_normalized[i, j]:.1f}%',
                           ha="center", va="center",
                           color="white" if cm_normalized[i, j] > 50 else "black",
                           fontsize=8)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Normalized %)')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存为 {save_path}")


# ==================== 可视化3：特征图 ====================
def plot_feature_maps(model, image, device, save_path='feature_maps.png'):
    """
    可视化 conv1 层的特征图。
    image: 已经过预处理的一个batch中的一张图，shape [1, 3, 224, 224]
    """
    model.eval()

    # 提取第一个卷积层的输出
    def hook_fn(module, input, output):
        hook_fn.features = output.detach()

    hook_fn.features = None
    hook = model.conv1.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(image.to(device))
    hook.remove()

    features = hook_fn.features[0].cpu()  # [64, H, W]

    # 画前16个通道
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < features.shape[0]:
            fmap = features[i].numpy()
            # 归一化到[0, 1]以便显示
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'Channel {i + 1}', fontsize=8)
        ax.axis('off')

    plt.suptitle('Feature Maps of conv1 (First 16 Channels)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"特征图已保存为 {save_path}")


# ==================== 主程序 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n" + "=" * 60)

    # 路径配置
    train_dir = r"data/mstar-train"
    test_dir = r"data/mstar-test"

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("路径不存在，请检查 train_dir / test_dir")
        return

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = MSTARDataset(train_dir, transform=train_transform, target_size=(224, 224))
    test_dataset = MSTARDataset(test_dir, transform=test_transform, target_size=(224, 224))
    train_dataset = get_stratified_subset(full_train_dataset, ratio=0.1, seed=GLOBAL_SEED)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # ==================== 官方预训练ResNet-18 ====================
    print("正在加载官方预训练ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    print("  预训练权重加载成功")

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()

    # 记录每一轮的指标
    train_losses, train_accs, test_accs = [], [], []
    model_save_path = Path(__file__).resolve().parent / 'best_limited_resnet18_mstar.pth'
    best_acc = 0.0
    patience = 5  # 定义耐心值：5epoch
    trigger_times = 0  # 定义触发次数

    for epoch in range(100):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        test_acc, test_preds, test_labels = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # +早停
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            trigger_times = 0  # 准确率提升，计数清零
            print(f"准确率提升! 最佳准确率: {best_acc:.2f}% (模型已保存)")
        else:
            trigger_times += 1
            print(f"耐心计数: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"\n早停触发! 训练结束。")
                break  # 跳出循环

        print(f"Epoch {epoch + 1:02d} | Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best:{best_acc:.2f}%\n")

    print(f"训练完成，最佳准确率: {best_acc:.2f}%")
    print("模型已保存为 best_limited_resnet18_mstar.pth")

    # ==================== 生成可视化 ====================
    print("\n正在生成可视化...")

    # 1. 训练曲线
    plot_training_curves(train_losses, train_accs, test_accs, best_acc, save_path='training_curves_10pct.png')

    # 2. 混淆矩阵
    print("加载最佳模型用于可视化...")
    model.load_state_dict(torch.load(model_save_path))
    _, final_preds, final_labels = evaluate(model, test_loader, device)
    plot_confusion_matrix(final_labels, final_preds, train_dataset.dataset.classes, save_path='confusion_matrix_10pct.png')

    # 3. 特征图
    test_dataset_no_aug = MSTARDataset(test_dir, transform=test_transform, target_size=(224, 224))
    sample_img, sample_label = test_dataset_no_aug[0]
    sample_img = sample_img.unsqueeze(0)  # [1, 3, 224, 224]
    print(f"特征图示例: 真实标签={test_dataset.classes[sample_label]}")
    plot_feature_maps(model, sample_img, device, save_path='feature_maps_10pct.png')

    print("\n所有可视化已生成完毕！")
    print("  - training_curves_10pct.png")
    print("  - confusion_matrix_10pct.png")
    print("  - feature_maps_10pct.png")


if __name__ == "__main__":
    main()
