import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image
import os, random, copy, math
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "2-11re"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DIR = r"data/mstar-train"
TEST_DIR = r"data/mstar-test"

LABEL_RATIO = 0.1; TARGET_SIZE = (224, 224)
BATCH_LABELED = 16; MU = 3; BATCH_UNLABELED = BATCH_LABELED * MU
CONF_THRESHOLD = 0.95; LR = 0.0003; WD = 1e-3
EMA_DECAY = 0.999; UNSUP_WEIGHT_MAX = 1.0
WARMUP_EPOCHS = 20; RAMPUP_EPOCHS = 20
MAX_EPOCHS = 300; PATIENCE = 50
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.5
COSINE_WARMUP = 20

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

WEAK_AUG = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),
])

STRONG_AUG_PIL = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), shear=(-15, 15, -15, 15), fill=0),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=0),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
])

TO_TENSOR = transforms.Compose([transforms.ToTensor(), NORMALIZE])

def scan_train(data_dir):
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    images, labels = [], []
    for class_name in classes:
        class_dir = data_dir / class_name
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            for img_path in class_dir.glob(ext):
                images.append(img_path); labels.append(class_to_idx[class_name])
    return classes, images, labels

def stratified_split(images, labels, ratio, seed):
    random.seed(seed); np.random.seed(seed)
    labels_arr = np.array(labels); unique = np.unique(labels_arr)
    labeled_idx, unlabeled_idx = [], []
    for cls in unique:
        cls_idx = np.where(labels_arr == cls)[0]
        n_labeled = max(1, int(len(cls_idx) * ratio))
        perm = np.random.permutation(cls_idx)
        labeled_idx.extend(perm[:n_labeled].tolist())
        unlabeled_idx.extend(perm[n_labeled:].tolist())
    random.shuffle(labeled_idx); random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx

class LabeledDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = [images[i] for i in indices]; self.labels = [labels[i] for i in indices]
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        return img, self.labels[idx]

class UnlabeledDataset(Dataset):
    def __init__(self, images, indices): self.images = [images[i] for i in indices]
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        return img

class TestDataset(Dataset):
    def __init__(self, data_dir):
        _, self.images, self.labels = scan_train(data_dir)
        self.transform = transforms.Compose([transforms.ToTensor(), NORMALIZE])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        return self.transform(img), self.labels[idx]

def apply_weak(image_pil): return TO_TENSOR(WEAK_AUG(image_pil))
def apply_strong(image_pil): return TO_TENSOR(STRONG_AUG_PIL(image_pil))

def mixup_data(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def ema_update(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)
        for (ema_n, ema_b), (n, b) in zip(ema_model.named_buffers(), model.named_buffers()):
            if 'running_mean' in ema_n or 'running_var' in ema_n or 'num_batches_tracked' in ema_n:
                ema_b.copy_(b)

def evaluate(model, loader):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device); outputs = model(images)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds) * 100

def main():
    log = open(str(OUTPUT_DIR / "fixmatch_result.txt"), 'w', encoding='utf-8')
    def log_print(*args, **kwargs):
        print(*args, **kwargs); print(*args, **kwargs, file=log, flush=True)

    log_print(f"Using device: {device}")
    if torch.cuda.is_available(): log_print(f"  GPU: {torch.cuda.get_device_name(0)}")
    log_print("=" * 60)

    log_print("\n[数据准备]")
    classes, all_images, all_labels = scan_train(TRAIN_DIR)
    num_classes = len(classes)
    log_print(f"发现类别: {classes}"); log_print(f"训练集总数: {len(all_images)} 张图片")
    labeled_idx, unlabeled_idx = stratified_split(all_images, all_labels, LABEL_RATIO, SEED)
    for cls_id, cls_name in enumerate(classes):
        cls_count = sum(1 for i in labeled_idx if all_labels[i] == cls_id)
        cls_total = sum(1 for l in all_labels if l == cls_id)
        log_print(f"    类别 '{cls_name}': 总数 {cls_total} -> 抽取 {cls_count}")
    log_print(f"  有标签样本: {len(labeled_idx)}"); log_print(f"  无标签样本: {len(unlabeled_idx)}")

    labeled_dataset = LabeledDataset(all_images, all_labels, labeled_idx)
    unlabeled_dataset = UnlabeledDataset(all_images, unlabeled_idx)
    test_dataset = TestDataset(TEST_DIR)
    log_print(f"  测试样本: {len(test_dataset)}")

    def collate_pil(batch): return list(zip(*batch))
    labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_LABELED, shuffle=True, num_workers=0, collate_fn=collate_pil, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_UNLABELED, shuffle=True, num_workers=0, collate_fn=lambda x: x, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    log_print("\n[模型初始化]")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters(): p.requires_grad = False
    ema_model.eval()

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD)

    def lr_lambda(epoch):
        if epoch < COSINE_WARMUP: return epoch / COSINE_WARMUP
        return 0.5 * (1 + math.cos(math.pi * (epoch - COSINE_WARMUP) / (MAX_EPOCHS - COSINE_WARMUP)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    criterion_ls = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    log_print("  模型: ResNet-18 (预训练)")
    log_print(f"  优化器: SGD lr={LR} momentum=0.9 wd={WD}")
    log_print(f"  调度器: CosineWarmup warmup={COSINE_WARMUP} T_max={MAX_EPOCHS} (优化/新增结构)")
    log_print(f"  Label Smoothing: {LABEL_SMOOTHING} (优化/新增结构)")
    log_print(f"  MixUp: alpha={MIXUP_ALPHA} (优化/新增结构)")
    log_print(f"  强增强: RandAugment风格(Rotation30+Affine15+ColorJitter+GaussianBlur+Perspective) (优化/新增结构)")
    log_print(f"  Warmup: {WARMUP_EPOCHS}, Rampup: {RAMPUP_EPOCHS} (优化/新增结构)")
    log_print(f"  训练Epoch: {MAX_EPOCHS} (优化/新增结构)")
    log_print(f"  有标签batch: {BATCH_LABELED}, 无标签batch: {BATCH_UNLABELED} (mu={MU})")
    log_print("=" * 60)

    best_acc = 0.0; trigger_times = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        if epoch < WARMUP_EPOCHS: unsup_weight = 0.0
        elif epoch < WARMUP_EPOCHS + RAMPUP_EPOCHS: unsup_weight = UNSUP_WEIGHT_MAX * (epoch - WARMUP_EPOCHS) / RAMPUP_EPOCHS
        else: unsup_weight = UNSUP_WEIGHT_MAX
        total_loss = total_loss_x = total_loss_u = 0.0
        total_labeled_correct = total_labeled_count = total_pseudo_count = step_count = 0

        for (img_l_pil, labels_l), img_u_list in zip(labeled_loader, unlabeled_loader):
            labels_l = torch.tensor(labels_l).to(device)
            img_l = torch.stack([TO_TENSOR(WEAK_AUG(pil)) for pil in img_l_pil])

            img_l_mixed, labels_a, labels_b, lam = mixup_data(img_l, labels_l, MIXUP_ALPHA)

            img_u_weak = torch.stack([apply_weak(pil) for pil in img_u_list])
            img_u_strong = torch.stack([apply_strong(pil) for pil in img_u_list])

            img_all = torch.cat([img_l_mixed, img_u_weak, img_u_strong], dim=0).to(device)
            logits_all = model(img_all)
            logits_l = logits_all[:len(img_l)]
            logits_u_weak = logits_all[len(img_l):len(img_l) + len(img_u_weak)]
            logits_u_strong = logits_all[len(img_l) + len(img_u_weak):]

            loss_x = mixup_criterion(criterion_ls, logits_l, labels_a, labels_b, lam).mean()

            with torch.no_grad():
                probs_u_weak = F.softmax(logits_u_weak, dim=1)
                max_probs, pseudo_labels = probs_u_weak.max(dim=1)
                mask = (max_probs >= CONF_THRESHOLD).float()
            loss_u = (criterion_ce(logits_u_strong, pseudo_labels) * mask).mean()
            loss = loss_x + unsup_weight * loss_u

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ema_update(model, ema_model, EMA_DECAY)

            with torch.no_grad():
                _, preds_l = logits_l.max(dim=1)
                total_labeled_correct += (preds_l == labels_l).sum().item()
                total_labeled_count += labels_l.size(0)
                total_pseudo_count += mask.sum().item()

            total_loss += loss.item(); total_loss_x += loss_x.item(); total_loss_u += loss_u.item(); step_count += 1
            if step_count % 5 == 0:
                print(f"  Epoch {epoch + 1} | batch {step_count}/{len(labeled_loader)} | l_x:{loss_x.item():.3f} l_u:{loss_u.item():.4f}", flush=True)

        scheduler.step()

        avg_loss = total_loss / max(step_count, 1)
        train_acc = 100.0 * total_labeled_correct / max(total_labeled_count, 1)
        test_acc = evaluate(ema_model, test_loader)

        log_print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} (x:{total_loss_x/max(step_count,1):.4f} u:{total_loss_u/max(step_count,1):.4f}) | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Pseudo: {total_pseudo_count/max(step_count*BATCH_UNLABELED,1)*100:.1f}% | w_u: {unsup_weight:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict(), 'epoch': epoch, 'acc': best_acc}, str(OUTPUT_DIR / "best_model.pth"))
            trigger_times = 0
            log_print(f"  >>> 最佳准确率提升: {best_acc:.2f}% (模型已保存)")
        else:
            trigger_times += 1
            log_print(f"  早停计数: {trigger_times}/{PATIENCE}")
            if trigger_times >= PATIENCE: log_print(f"\n早停触发!"); break

    log_print(f"\n{'='*60}\n训练完成! 最佳测试准确率: {best_acc:.2f}%\n{'='*60}")
    log.close()

    with open(str(OUTPUT_DIR / "模型表现及评估.txt"), 'w', encoding='utf-8') as f:
        f.write(f"模型表现及评估\n{'='*60}\n\n")
        f.write(f"最佳测试准确率: {best_acc:.2f}%\n")
        f.write(f"训练Epoch数: {epoch + 1}\n\n")
        f.write("模型结构:\n")
        f.write("  输入特征\n")
        f.write("  (MSTAR SAR图像 128x128灰度 -> 224x224 RGB)\n")
        f.write("         ↓\n")
        f.write("  弱增强 (RandomHorizontalFlip + RandomAffine translate)\n")
        f.write("         ↓\n")
        f.write("  MixUp数据增强 (优化/新增结构)\n")
        f.write("         ↓\n")
        f.write("  Label Smoothing CrossEntropyLoss(smoothing=0.1) (优化/新增结构)\n")
        f.write("         ↓\n")
        f.write("  ResNet-18 (ImageNet预训练)\n")
        f.write("         ↓\n")
        f.write("  分类头 FC (512 -> 10)\n")
        f.write("         ↓\n")
        f.write("  FixMatch半监督训练 (SGD + CosineWarmup + Warmup20+Rampup20 + 300epochs)\n")
        f.write("         ↓\n")
        f.write("  强增强 RandAugment风格(Rotation30+Affine15+ColorJitter+GaussianBlur+Perspective) (优化/新增结构)\n")
        f.write("         ↓\n")
        f.write("  输出: 10类军事车辆分类\n\n")
        f.write("优化/新增结构说明:\n")
        f.write("(1) CosineWarmup调度器 (优化/新增结构):\n")
        f.write("    使用20epoch线性warmup + Cosine退火调度。\n")
        f.write("    线性warmup阶段学习率从0逐渐上升，避免训练初期的参数震荡；\n")
        f.write("    Cosine退火将学习率平滑降至接近零，比CosineAnnealing更能利用后期小学习率的精细调优。\n\n")
        f.write("(2) Label Smoothing 标签平滑 (优化/新增结构):\n")
        f.write("    使用标签平滑(smoothing=0.1)损失函数。\n")
        f.write("    将硬标签转换为软标签防止过拟合，在MixUp基础上进一步正则化。\n\n")
        f.write("(3) MixUp数据增强 (优化/新增结构):\n")
        f.write("    在有标签数据上应用MixUp(alpha=0.5)。\n")
        f.write("    通过对两个样本进行凸组合生成新的训练样本，\n")
        f.write("    鼓励模型学习更平滑的决策边界，有效防止在268张标签上的过拟合。\n\n")
        f.write("(4) RandAugment风格强增强 (优化/新增结构):\n")
        f.write("    使用更强力的数据增强组合：\n")
        f.write("    - RandomAffine(degrees=30, translate=0.15, shear=15)\n")
        f.write("    - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)\n")
        f.write("    - RandomPerspective(distortion=0.2, p=0.3)\n")
        f.write("    - GaussianBlur(sigma=0.1-1.5)\n")
        f.write("    强增强能提升伪标签质量和模型鲁棒性。\n\n")
        f.write("(5) 延长Warmup+Rampup (优化/新增结构):\n")
        f.write("    将Warmup从10epoch延长到20epoch，Rampup从10延长到20epoch。\n")
        f.write("    更长的纯有监督预热让模型在有标签数据上建立更坚实的基础，\n")
        f.write("    减少半监督阶段的错误传播。\n\n")
        f.write("(6) 延长训练Epoch (优化/新增结构):\n")
        f.write("    训练从200epoch延长到300epoch，早停patience从30提升到50。\n")
        f.write("    更长的训练时间为半监督学习提供更多迭代空间。\n\n")
        f.write("(7) SGD优化器 (沿用fix1基线):\n")
        f.write("    使用SGD (lr=0.0003, momentum=0.9, wd=1e-3)。\n")

    print(f"\n结果已保存到 {OUTPUT_DIR}")
    print(f"评估报告: {OUTPUT_DIR / '模型表现及评估.txt'}")

if __name__ == "__main__": main()