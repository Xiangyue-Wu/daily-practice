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


# ==================== 分层抽样 ====================
def get_stratified_subset(full_dataset, ratio, seed):
    random.seed(seed)
    np.random.seed(seed)

    targets = np.array(full_dataset.labels)
    classes = np.unique(targets)
    selected_indices = []

    print(f"\n--- 正在执行分层抽样 (比例: {ratio}, 种子: {seed}) ---")
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        num_samples = max(1, int(len(cls_indices) * ratio))  # 至少保留1个样本
        num_samples = min(len(cls_indices), num_samples)
        sampled = np.random.choice(cls_indices, num_samples, replace=False)
        selected_indices.extend(sampled.tolist())
        print(f"  类别 '{full_dataset.classes[cls]}': 总数 {len(cls_indices)} -> 抽取 {num_samples}")

    random.shuffle(selected_indices)
    print(f"已抽样完成！共选取 {len(selected_indices)} 张图片用于训练。\n" + "-" * 60)

    return Subset(full_dataset, selected_indices)


# ==================== 训练与评估 ====================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, verbose=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        if verbose and batch_idx % 10 == 0:
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


# ==================== 可视化1：训练曲线（单模型）====================
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
    可视化 conv1 层的特征图
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
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'Channel {i + 1}', fontsize=8)
        ax.axis('off')

    plt.suptitle('Feature Maps of conv1 (First 16 Channels)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"特征图已保存为 {save_path}")


# ==================== 可视化4：迭代精度对比曲线（新增）====================
def plot_comparison_curves(full_train_losses, full_train_accs, full_test_accs,
                           limited_train_losses, limited_train_accs, limited_test_accs,
                           best_full_acc, best_limited_acc,
                           save_path='accuracy_comparison.png'):
    """
    绘制全量标签 vs 有限标签的训练对比曲线
    """
    epochs_full = range(1, len(full_train_losses) + 1)
    epochs_limited = range(1, len(limited_train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss对比曲线
    axes[0].plot(epochs_full, full_train_losses, 'b-', label='Full Labels (100%)', linewidth=1.5)
    axes[0].plot(epochs_limited, limited_train_losses, 'r-', label='Limited Labels (10%)', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 训练准确率对比曲线
    axes[1].plot(epochs_full, full_train_accs, 'b-', label='Full Labels (100%)', linewidth=1.5)
    axes[1].plot(epochs_limited, limited_train_accs, 'r-', label='Limited Labels (10%)', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Train Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 测试准确率对比曲线（核心对比）
    axes[2].plot(epochs_full, full_test_accs, 'b-', label='Full Labels (100%)', linewidth=1.5)
    axes[2].plot(epochs_limited, limited_test_accs, 'r-', label='Limited Labels (10%)', linewidth=1.5)
    axes[2].axhline(y=best_full_acc, color='b', linestyle='--', alpha=0.5, label=f'Full Best = {best_full_acc:.2f}%')
    axes[2].axhline(y=best_limited_acc, color='r', linestyle='--', alpha=0.5,
                    label=f'Limited Best = {best_limited_acc:.2f}%')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Test Accuracy Comparison')
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.3)

    # 添加性能差距标注
    diff = best_full_acc - best_limited_acc
    axes[2].text(0.02, 0.02, f'Performance Gap: {diff:.2f}%',
                 transform=axes[2].transAxes, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.suptitle('Comparison: Full Labels (100%) vs Limited Labels (10%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"迭代精度对比曲线已保存为 {save_path}")


# ==================== 可视化5：柱状图对比 ====================
def plot_bar_comparison(full_acc, limited_acc, save_path='bar_comparison.png'):
    """
    绘制柱状图对比全量标签和有限标签的最佳准确率
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    categories = ['Full Labels (100%)', 'Limited Labels (10%)']
    accuracies = [full_acc, limited_acc]
    colors = ['#2E86AB', '#A23B72']

    bars = ax.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=1.5)

    # 在柱子上方显示数值
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 添加性能下降箭头
    diff = full_acc - limited_acc
    ax.annotate(f'↓ {diff:.2f}个百分点',
                xy=(1, limited_acc), xytext=(0.5, (full_acc + limited_acc) / 2),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2),
                fontsize=12, color='red', ha='center')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Best Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"柱状对比图已保存为 {save_path}")


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

    # ========== 第一部分：训练全量标签模型（100%）==========
    print("\n" + "=" * 60)
    print("第一部分：训练全量标签模型 (100% 数据)")
    print("=" * 60)

    full_train_dataset = MSTARDataset(train_dir, transform=train_transform, target_size=(224, 224))
    test_dataset = MSTARDataset(test_dir, transform=test_transform, target_size=(224, 224))

    full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 训练全量模型
    print("正在加载官方预训练ResNet-18...")
    full_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    full_model.fc = nn.Linear(full_model.fc.in_features, full_train_dataset.num_classes)
    full_model = full_model.to(device)

    full_optimizer = optim.SGD(full_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    full_scheduler = optim.lr_scheduler.CosineAnnealingLR(full_optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()

    full_train_losses, full_train_accs, full_test_accs = [], [], []
    best_full_acc = 0.0
    full_patience = 5
    full_trigger_times = 0

    print("开始训练全量模型...")
    for epoch in range(100):
        train_loss, train_acc = train_one_epoch(full_model, full_train_loader, criterion,
                                                full_optimizer, device, epoch, verbose=False)
        full_scheduler.step()
        test_acc, _, _ = evaluate(full_model, test_loader, device)

        full_train_losses.append(train_loss)
        full_train_accs.append(train_acc)
        full_test_accs.append(test_acc)

        if test_acc > best_full_acc:
            best_full_acc = test_acc
            torch.save(full_model.state_dict(), 'result/2/best_full_resnet18_mstar2.pth')
            full_trigger_times = 0
            print(f"✓ 全量模型准确率提升! 最佳: {best_full_acc:.2f}%")
        else:
            full_trigger_times += 1
            if full_trigger_times >= full_patience:
                print(f"\n早停触发! 全量模型训练结束。")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:02d} | Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    print(f"\n全量模型训练完成，最佳准确率: {best_full_acc:.2f}%")

    # ========== 第二部分：训练有限标签模型（10%）==========
    print("\n" + "=" * 60)
    print("第二部分：训练有限标签模型 (10% 数据)")
    print("=" * 60)

    # 重新加载数据集（避免缓存问题）
    full_train_dataset_fresh = MSTARDataset(train_dir, transform=train_transform, target_size=(224, 224))
    train_subset = get_stratified_subset(full_train_dataset_fresh, ratio=0.1, seed=GLOBAL_SEED)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)

    # 重新初始化模型
    limited_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    limited_model.fc = nn.Linear(limited_model.fc.in_features, full_train_dataset_fresh.num_classes)
    limited_model = limited_model.to(device)

    limited_optimizer = optim.SGD(limited_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    limited_scheduler = optim.lr_scheduler.CosineAnnealingLR(limited_optimizer, T_max=30)

    limited_train_losses, limited_train_accs, limited_test_accs = [], [], []
    best_limited_acc = 0.0
    limited_patience = 5
    limited_trigger_times = 0

    print("开始训练有限标签模型...")
    for epoch in range(100):
        train_loss, train_acc = train_one_epoch(limited_model, train_loader, criterion,
                                                limited_optimizer, device, epoch, verbose=False)
        limited_scheduler.step()
        test_acc, _, _ = evaluate(limited_model, test_loader, device)

        limited_train_losses.append(train_loss)
        limited_train_accs.append(train_acc)
        limited_test_accs.append(test_acc)

        if test_acc > best_limited_acc:
            best_limited_acc = test_acc
            torch.save(limited_model.state_dict(), 'result/2/best_limited_resnet18_mstar2.pth')
            limited_trigger_times = 0
            print(f"✓ 有限标签模型准确率提升! 最佳: {best_limited_acc:.2f}%")
        else:
            limited_trigger_times += 1
            if limited_trigger_times >= limited_patience:
                print(f"\n早停触发! 有限标签模型训练结束。")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:02d} | Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    print(f"\n有限标签模型训练完成，最佳准确率: {best_limited_acc:.2f}%")

    # ==================== 生成对比可视化 ====================
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    # 1. 迭代精度对比曲线（三子图）
    plot_comparison_curves(
        full_train_losses, full_train_accs, full_test_accs,
        limited_train_losses, limited_train_accs, limited_test_accs,
        best_full_acc, best_limited_acc,
        save_path='result/2/accuracy_comparison_full_vs_limited.png'
    )

    # 2. 柱状图对比
    plot_bar_comparison(best_full_acc, best_limited_acc, save_path='result/2/bar_comparison.png')

    # 3. 有限标签模型的训练曲线
    plot_training_curves(limited_train_losses, limited_train_accs, limited_test_accs,
                         best_limited_acc, save_path='result/2/training_curves_10pct.png')

    # 4. 混淆矩阵（有限标签）
    print("加载最佳有限标签模型生成混淆矩阵...")
    limited_model.load_state_dict(torch.load('best_limited_resnet18_mstar.pth'))
    _, final_preds, final_labels = evaluate(limited_model, test_loader, device)
    plot_confusion_matrix(final_labels, final_preds, test_dataset.classes,
                          save_path='result/2/confusion_matrix_10pct.png')

    # 5. 特征图（有限标签）
    test_dataset_no_aug = MSTARDataset(test_dir, transform=test_transform, target_size=(224, 224))
    sample_img, sample_label = test_dataset_no_aug[0]
    sample_img = sample_img.unsqueeze(0)
    print(f"特征图示例: 真实标签={test_dataset.classes[sample_label]}")
    plot_feature_maps(limited_model, sample_img, device, save_path='result/2/feature_maps_10pct.png')

    # ==================== 打印最终结果 ====================
    print("\n" + "=" * 60)
    print("最终性能对比结果")
    print("=" * 60)
    print(f"全量标签模型 (100% 数据): 最佳测试准确率 = {best_full_acc:.2f}%")
    print(f"有限标签模型 (10% 数据):  最佳测试准确率 = {best_limited_acc:.2f}%")
    print(f"性能差距: {best_full_acc - best_limited_acc:.2f} 个百分点")
    print(f"性能衰减率: {(best_full_acc - best_limited_acc) / best_full_acc * 100:.2f}%")
    print("=" * 60)

    print("\n所有可视化已生成完毕！")
    print("生成的文件列表：")
    print("  1. accuracy_comparison_full_vs_limited.png  - 迭代精度对比曲线（Loss/训练acc/测试acc）")
    print("  2. bar_comparison.png                      - 柱状图对比")
    print("  3. training_curves_10pct.png               - 有限标签训练曲线")
    print("  4. confusion_matrix_10pct.png              - 混淆矩阵")
    print("  5. feature_maps_10pct.png                  - 特征图可视化")
    print("\n模型权重文件：")
    print("  - best_full_resnet18_mstar.pth")
    print("  - best_limited_resnet18_mstar.pth")


if __name__ == "__main__":
    main()