import os
import time
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_RATIO = 1.0
TEST_RATIO = 1.0

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


__all__ = ["ResNetCifar", "resnet32", "train_model", "evaluate_model", "main"]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None


def conv3x3(in_plane, out_plane, stride=1):
    return nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, norm_layer=nn.BatchNorm2d):
        super(ResNetCifar, self).__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3, 16, stride=1)
        self.bn1 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1, None, norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet32(num_classes=10):
    model = ResNetCifar(BasicBlock, [5, 5, 5], num_classes=num_classes)
    return model


def get_cifar10_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return train_transform, test_transform


def detect_dataset_type(data_root):
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        return "imagefolder"
    return "cifar10"


def build_dataloader(data_root, batch_size, num_workers, download, train_ratio=0.0001, test_ratio=0.0001):
    train_transform, test_transform = get_cifar10_transforms()
    dataset_type = detect_dataset_type(data_root)

    if dataset_type == "imagefolder":
        train_dir = os.path.join(data_root, "train")
        test_dir = os.path.join(data_root, "test")
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
        class_names = train_dataset.classes
    else:
        train_dataset = datasets.CIFAR10(root=data_root, train=True, transform=train_transform, download=download)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=test_transform, download=download)
        class_names = train_dataset.classes

    pin_memory = torch.cuda.is_available()

    train_subset_size = int(len(train_dataset) * train_ratio)
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    train_indices = train_indices[:train_subset_size]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    test_subset_size = int(len(test_dataset) * test_ratio)
    test_indices = list(range(len(test_dataset)))
    test_subset = torch.utils.data.Subset(test_dataset, test_indices[:test_subset_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, class_names


def prepare_result_dir(result_dir):
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_show_and_close(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print("可视化结果已保存:", save_path)
    plt.show()
    plt.close(fig)
    return None


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_num += images.size(0)

    avg_loss = total_loss / total_num
    avg_acc = 100.0 * total_correct / total_num
    return avg_loss, avg_acc


def build_confusion_matrix(true_labels, pred_labels, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion_matrix[true_label, pred_label] += 1
    return confusion_matrix


def denormalize_image(image_tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    image = torch.clamp(image, 0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    return image


def collect_prediction_results(model, data_loader, device, sample_count=16):
    model.eval()
    true_labels = []
    pred_labels = []
    sample_images = []
    sample_true_labels = []
    sample_pred_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            cached_images = images.cpu()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())

            remain_count = sample_count - len(sample_images)
            if remain_count > 0:
                sample_images.extend(list(cached_images[:remain_count]))
                sample_true_labels.extend(labels[:remain_count].cpu().tolist())
                sample_pred_labels.extend(preds[:remain_count].cpu().tolist())

    return true_labels, pred_labels, sample_images, sample_true_labels, sample_pred_labels


def plot_loss_curve(history, result_dir):
    if len(history["epoch"]) == 0:
        print("未生成损失曲线：当前没有训练历史。")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", linewidth=2, label="训练损失")
    ax.plot(history["epoch"], history["test_loss"], marker="s", linewidth=2, label="测试损失")
    ax.set_title("ResNet32 在 CIFAR-10 上的损失曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    save_path = os.path.join(result_dir, "loss_curve.png")
    save_show_and_close(fig, save_path)
    return None


def plot_accuracy_curve(history, result_dir):
    if len(history["epoch"]) == 0:
        print("未生成准确率曲线：当前没有训练历史。")
        return None

    best_index = int(np.argmax(history["test_acc"]))
    best_epoch = history["epoch"][best_index]
    best_acc = history["test_acc"][best_index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_acc"], marker="o", linewidth=2, label="训练准确率")
    ax.plot(history["epoch"], history["test_acc"], marker="s", linewidth=2, label="测试准确率")
    ax.scatter(best_epoch, best_acc, color="red", s=80, label="最佳测试准确率")
    ax.annotate(
        "最佳: {:.2f}%".format(best_acc),
        xy=(best_epoch, best_acc),
        xytext=(best_epoch, best_acc + 2),
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_title("ResNet32 在 CIFAR-10 上的准确率曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    save_path = os.path.join(result_dir, "accuracy_curve.png")
    save_show_and_close(fig, save_path)
    return None


def plot_confusion_matrix(true_labels, pred_labels, class_names, result_dir):
    confusion_matrix = build_confusion_matrix(true_labels, pred_labels, len(class_names))

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("ResNet32 在 CIFAR-10 测试集上的混淆矩阵")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    threshold = confusion_matrix.max() / 2.0 if confusion_matrix.max() > 0 else 0.0
    for row_index in range(confusion_matrix.shape[0]):
        for col_index in range(confusion_matrix.shape[1]):
            color = "white" if confusion_matrix[row_index, col_index] > threshold else "black"
            ax.text(
                col_index,
                row_index,
                str(confusion_matrix[row_index, col_index]),
                ha="center",
                va="center",
                color=color,
            )

    save_path = os.path.join(result_dir, "confusion_matrix.png")
    save_show_and_close(fig, save_path)
    return None


def plot_class_accuracy_bar(true_labels, pred_labels, class_names, result_dir):
    confusion_matrix = build_confusion_matrix(true_labels, pred_labels, len(class_names))
    class_totals = confusion_matrix.sum(axis=1)
    class_correct = np.diag(confusion_matrix)
    class_accuracy = np.divide(
        class_correct * 100.0,
        class_totals,
        out=np.zeros_like(class_correct, dtype=np.float64),
        where=class_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, class_accuracy, color="skyblue", edgecolor="black")
    ax.set_title("ResNet32 在 CIFAR-10 各类别上的测试准确率")
    ax.set_xlabel("类别")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for bar, acc_value in zip(bars, class_accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            "{:.1f}".format(acc_value),
            ha="center",
            va="bottom",
        )

    save_path = os.path.join(result_dir, "class_accuracy_bar.png")
    save_show_and_close(fig, save_path)
    return None


def plot_sample_predictions(sample_images, sample_true_labels, sample_pred_labels, class_names, result_dir):
    if len(sample_images) == 0:
        print("未生成测试样本预测图：当前没有可用样本。")
        return None

    show_count = min(len(sample_images), 16)
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    for index, ax in enumerate(axes):
        if index < show_count:
            image = denormalize_image(sample_images[index])
            true_name = class_names[sample_true_labels[index]]
            pred_name = class_names[sample_pred_labels[index]]
            title_color = "green" if sample_true_labels[index] == sample_pred_labels[index] else "red"
            ax.imshow(image)
            ax.set_title("真:{}\n预测:{}".format(true_name, pred_name), color=title_color, fontsize=10)
        ax.axis("off")

    fig.suptitle("ResNet32 在 CIFAR-10 测试集上的样本预测示例", fontsize=14)
    save_path = os.path.join(result_dir, "sample_predictions.png")
    save_show_and_close(fig, save_path)
    return None


def train_model(model, train_loader, test_loader, device, epochs, lr, weight_decay, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    best_acc = 0.0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_num = 0
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_num += images.size(0)

        scheduler.step()

        train_loss = total_loss / total_num
        train_acc = 100.0 * total_correct / total_num
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        epoch_time = time.time() - start_time
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

        print(
            "Epoch [{:03d}/{:03d}] "
            "Train Loss: {:.4f} Train Acc: {:.2f}% "
            "Test Loss: {:.4f} Test Acc: {:.2f}% "
            "Best: {:.2f}% Time: {:.2f}s".format(
                epoch + 1, epochs, train_loss, train_acc, test_loss, test_acc, best_acc, epoch_time
            )
        )

    return best_acc, history


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet32 training and testing on CIFAR-10")
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"data/cifar10",
        help="CIFAR-10 data root path",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=0, help="Data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--download", action="store_true", help="Download dataset automatically")
    parser.add_argument(
        "--save_path",
        type=str,
        default=r"models/resnet32_cifar10_best.pth",
        help="Path for saving best model",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=r"outputs/resnet32_cifar10_results",
        help="Directory for saving visualization results",
    )
    parser.add_argument("--only_test", action="store_true", help="Only test by loading the saved checkpoint")
    args = parser.parse_args()
    return args


def main():
    # 1. 参数解析、随机种子与设备设置
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = prepare_result_dir(args.result_dir)
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # 2. 数据集加载与模型构建
    train_loader, test_loader, class_names = build_dataloader(
        args.data_root, args.batch_size, args.num_workers, args.download, TRAIN_RATIO, TEST_RATIO
    )
    model = resnet32(num_classes=len(class_names)).to(device)
    print("Use device:", device)
    print("Dataset root:", args.data_root)
    print("Classes:", class_names)
    print("Number of classes:", len(class_names))
    print("Train samples:", len(train_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Model parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    print("Visualization result directory:", result_dir)

    # 3. 若最优模型已存在则跳过训练直接加载，否则开始训练
    if os.path.exists(args.save_path):
        print("最优模型已存在，跳过训练，直接加载: {}".format(args.save_path))
    else:
        best_acc, history = train_model(
            model,
            train_loader,
            test_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.save_path,
        )
        print("Training finished, best test accuracy: {:.2f}%".format(best_acc))

    # 4. 加载最佳权重并在测试集上评估
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print("Final Test Loss: {:.4f}, Final Test Acc: {:.2f}%".format(test_loss, test_acc))

        # 5. 收集测试集预测结果，供后续多个可视化函数复用
        true_labels, pred_labels, sample_images, sample_true_labels, sample_pred_labels = collect_prediction_results(
            model, test_loader, device
        )

        # 6. 绘制损失曲线，不需要时可删除本行调用
        plot_loss_curve(history, result_dir)

        # 7. 绘制准确率曲线，不需要时可删除本行调用
        plot_accuracy_curve(history, result_dir)

        # 8. 绘制混淆矩阵，不需要时可删除本行调用
        plot_confusion_matrix(true_labels, pred_labels, class_names, result_dir)

        # 9. 绘制分类别准确率柱状图，不需要时可删除本行调用
        plot_class_accuracy_bar(true_labels, pred_labels, class_names, result_dir)

        # 10. 绘制测试样本预测图，不需要时可删除本行调用
        plot_sample_predictions(sample_images, sample_true_labels, sample_pred_labels, class_names, result_dir)
    else:
        print("Checkpoint not found:", args.save_path)

    return None


if __name__ == "__main__":
    main()

