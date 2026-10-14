import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import os, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "comparison_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DIR = r"data/mstar-test"
TARGET_SIZE = (224, 224)

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), NORMALIZE])

ORIGINAL_MODEL_PATH = BASE_DIR / "best_limited_resnet18_mstar.pth"
IMPROVED_MODEL_PATH = BASE_DIR / "2-11re" / "best_model.pth"
IMPROVED_LOG_PATH = BASE_DIR / "2-11re" / "fixmatch_result.txt"


def scan_classes(data_dir):
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    images, labels = [], []
    for class_name in classes:
        class_dir = data_dir / class_name
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            for img_path in class_dir.glob(ext):
                images.append(img_path)
                labels.append(classes.index(class_name))
    return classes, images, labels


class TestDataset(Dataset):
    def __init__(self, data_dir):
        _, self.images, self.labels = scan_classes(data_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        return TEST_TRANSFORM(img), self.labels[idx]


def load_original_model(path, num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model' not in state and 'ema_model' not in state:
        model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_improved_model(path, num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['ema_model'])
    model.to(device)
    model.eval()
    return model


def evaluate(model, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds) * 100, all_preds, all_labels


def parse_fixmatch_log(log_path):
    """
    Parse fixmatch_result.txt and extract training metrics.
    Returns dict with keys: epoch, loss, loss_x, loss_u, train_acc, test_acc,
                             pseudo_ratio, unsup_weight, lr
    """
    pattern = re.compile(
        r'Epoch\s+(\d+)\s*\|\s*Loss:\s+([\d.]+)\s*\(x:([\d.]+)\s*u:([\d.]+)\)\s*\|'
        r'\s*Train:\s+([\d.]+)%\s*\|\s*Test:\s+([\d.]+)%\s*\|\s*Pseudo:\s+([\d.]+)%\s*\|'
        r'\s*w_u:\s+([\d.]+)\s*\|\s*LR:\s+([\d.]+)'
    )

    metrics = {
        'epoch': [], 'loss': [], 'loss_x': [], 'loss_u': [],
        'train_acc': [], 'test_acc': [], 'pseudo_ratio': [],
        'unsup_weight': [], 'lr': []
    }

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                metrics['epoch'].append(int(m.group(1)))
                metrics['loss'].append(float(m.group(2)))
                metrics['loss_x'].append(float(m.group(3)))
                metrics['loss_u'].append(float(m.group(4)))
                metrics['train_acc'].append(float(m.group(5)))
                metrics['test_acc'].append(float(m.group(6)))
                metrics['pseudo_ratio'].append(float(m.group(7)))
                metrics['unsup_weight'].append(float(m.group(8)))
                metrics['lr'].append(float(m.group(9)))
    return metrics


def plot_improved_loss_curves(metrics, save_path):
    epochs = metrics['epoch']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, metrics['loss'], 'k-', label='Total Loss', linewidth=1.2)
    axes[0].plot(epochs, metrics['loss_x'], 'b-', label='Labeled Loss (x)', linewidth=1.0, alpha=0.7)
    axes[0].plot(epochs, metrics['loss_u'], 'r-', label='Unlabeled Loss (u)', linewidth=1.0, alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('FixMatch Training Loss (Improved Model)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, metrics['train_acc'], 'b-', label='Train Acc (labeled)', linewidth=1.2)
    axes[1].plot(epochs, metrics['test_acc'], 'r-', label='Test Acc', linewidth=1.5)
    best_idx = np.argmax(metrics['test_acc'])
    best_acc = metrics['test_acc'][best_idx]
    best_ep = metrics['epoch'][best_idx]
    axes[1].axhline(y=best_acc, color='g', linestyle='--',
                    label=f'Best Test Acc = {best_acc:.2f}% (Epoch {best_ep})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('FixMatch Accuracy (Improved Model)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 改进模型损失/精度曲线: {save_path}")


def plot_pseudo_label_curve(metrics, save_path):
    epochs = metrics['epoch']
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pseudo Label Ratio (%)', color=color1)
    ax1.plot(epochs, metrics['pseudo_ratio'], 'b-', label='Pseudo Label Ratio', linewidth=1.2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Unsupervised Weight', color=color2)
    ax2.plot(epochs, metrics['unsup_weight'], 'r--', label='Unsup Weight (w_u)', linewidth=1.2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    ax1.set_title('Pseudo Label Ratio & Unsupervised Weight')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 伪标签比率曲线: {save_path}")


def plot_confusion_matrix_img(all_labels, all_preds, class_names, title, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm_norm[i, j]:.1f}%', ha="center", va="center",
                    color="white" if cm_norm[i, j] > 50 else "black", fontsize=8)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 混淆矩阵: {save_path}")


def plot_accuracy_recovery(orig_acc, improved_acc, class_names, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [orig_acc, improved_acc], width,
                   color=['#E74C3C', '#2ECC71'], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy Recovery: 1/10 Supervised vs FixMatch Semi-Supervised')
    ax.set_xticks(x)
    ax.set_xticklabels(['Original Model\n(1/10 Fully Supervised)', 'Improved Model\n(FixMatch Semi-Supervised)'])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars1, [orig_acc, improved_acc]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.0,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    improvement = improved_acc - orig_acc
    ax.annotate(f'+{improvement:.2f}%', xy=(1, improved_acc),
                xytext=(1, improved_acc + 8), ha='center', fontsize=11,
                color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 精度恢复对比图: {save_path}")


def main():
    print("=" * 60)
    print(f"模型对比分析 | 设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    print("\n[1] 加载测试集...")
    test_dataset = TestDataset(TEST_DIR)
    classes, _, _ = scan_classes(TEST_DIR)
    print(f"  类别 ({len(classes)}): {classes}")
    print(f"  测试样本: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    print("\n[2] 加载改进模型 (FixMatch ResNet-18 + EMA)...")
    improved_model = load_improved_model(IMPROVED_MODEL_PATH)
    print(f"  模型路径: {IMPROVED_MODEL_PATH}")

    print("\n[3] 加载原模型 (1/10 Fully Supervised ResNet-18)...")
    original_model = load_original_model(ORIGINAL_MODEL_PATH)
    print(f"  模型路径: {ORIGINAL_MODEL_PATH}")

    print("\n[4] 评估模型...")
    improved_acc, improved_preds, improved_labels = evaluate(improved_model, test_loader)
    print(f"  改进模型 (FixMatch) 测试准确率: {improved_acc:.2f}%")
    original_acc, original_preds, original_labels = evaluate(original_model, test_loader)
    print(f"  原模型 (1/10监督)  测试准确率: {original_acc:.2f}%")
    print(f"  精度提升: +{improved_acc - original_acc:.2f}%")

    print("\n[5] 解析改进模型训练日志...")
    metrics = parse_fixmatch_log(IMPROVED_LOG_PATH)
    print(f"  共解析 {len(metrics['epoch'])} 个epoch的训练数据")
    best_idx = np.argmax(metrics['test_acc'])
    print(f"  日志中最佳测试准确率: {metrics['test_acc'][best_idx]:.2f}% (Epoch {metrics['epoch'][best_idx]})")

    print("\n[6] 生成可视化...")

    plot_improved_loss_curves(metrics, str(OUTPUT_DIR / 'improved_loss_accuracy.png'))
    plot_pseudo_label_curve(metrics, str(OUTPUT_DIR / 'pseudo_label_curve.png'))

    plot_confusion_matrix_img(improved_labels, improved_preds, classes,
                              'Confusion Matrix - Improved Model (FixMatch)',
                              str(OUTPUT_DIR / 'confusion_matrix_improved.png'))
    plot_confusion_matrix_img(original_labels, original_preds, classes,
                              'Confusion Matrix - Original Model (1/10 Supervised)',
                              str(OUTPUT_DIR / 'confusion_matrix_original.png'))

    plot_accuracy_recovery(original_acc, improved_acc, classes,
                           str(OUTPUT_DIR / 'accuracy_recovery.png'))

    print("\n[7] 生成对比表格...")
    table_path = OUTPUT_DIR / 'accuracy_comparison.txt'
    with open(str(table_path), 'w', encoding='utf-8') as f:
        f.write("精度恢复情况对比分析\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试集: MSTAR/mstar-test ({len(test_dataset)} 张图片)\n")
        f.write(f"类别数: {len(classes)}\n\n")
        f.write("-" * 60 + "\n")
        imp_label = f"{improved_acc:.2f}%"
        orig_label = f"{original_acc:.2f}%"
        delta_label = f"+{improved_acc - original_acc:.2f}%"
        f.write(f"{'指标':<30} {'原模型(1/10监督)':<20} {'改进模型(FixMatch)':<20}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'测试准确率':<30} {orig_label:<20} {imp_label:<20}\n")
        f.write(f"{'精度提升':<30} {'--':<20} {delta_label:<20}\n")
        f.write("-" * 70 + "\n\n")

        f.write("说明:\n")
        f.write("  原模型: ResNet-18全监督训练, 仅使用1/10训练标签 (268张)\n")
        f.write("          SGD(lr=0.001, momentum=0.9, wd=5e-4), CosineAnnealing(T_max=30)\n")
        f.write("          100 epochs, early stopping(patience=5)\n\n")
        f.write("  改进模型: ResNet-18 FixMatch半监督训练, 268张标签 + 2478张无标签\n")
        f.write("            SGD(lr=0.0003, momentum=0.9, wd=1e-3), CosineWarmup(20+280)\n")
        f.write("            300 epochs, MixUp(alpha=0.5), LabelSmoothing(0.1)\n")
        f.write("            RandAugment强增强, Warmup+Rampup各20epoch\n\n")
        f.write("  精度恢复率: 半监督方法相较于仅用1/10标签的全监督方法,\n")
        f.write(f"  将准确率从 {original_acc:.2f}% 提升至 {improved_acc:.2f}%,\n")
        f.write(f"  提升幅度 {improved_acc - original_acc:.2f} 个百分点。\n")
    print(f"  [OK] 对比表格: {table_path}")

    print("\n" + "=" * 60)
    print(f"所有文件已保存到: {OUTPUT_DIR}")
    print(f"  - improved_loss_accuracy.png      (改进模型损失/精度曲线)")
    print(f"  - pseudo_label_curve.png          (伪标签比率曲线)")
    print(f"  - confusion_matrix_improved.png   (改进模型混淆矩阵)")
    print(f"  - confusion_matrix_original.png   (原模型混淆矩阵)")
    print(f"  - accuracy_recovery.png           (精度恢复对比图)")
    print(f"  - accuracy_comparison.txt         (精度对比表格)")
    print("=" * 60)


if __name__ == "__main__":
    main()
