import os
import time
import ast
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"data/cifar10"
RESULT_DIR = r"outputs/re-compare"

BASELINE_PATH = r"models/resnet32_cifar10_best.pth"
PSO_PATH = r"models/resnet32_pso_pruned.pth"
GA_PATH = r"models/ga-pruned_best.pth"

PSO_RESULT_TXT = r"outputs/pso_result.txt"
GA_RESULT_TXT = r"outputs/results.txt"


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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetCifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, norm_layer=nn.BatchNorm2d,
                 stage_channels=None):
        super(ResNetCifar, self).__init__()
        if stage_channels is None:
            stage_channels = [16, 32, 64]
        self.stage_channels = stage_channels
        self.in_planes = stage_channels[0]
        self.conv1 = conv3x3(3, stage_channels[0], stride=1)
        self.bn1 = norm_layer(stage_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, stage_channels[0], layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, stage_channels[1], layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, stage_channels[2], layers[2], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_channels[2] * block.expansion, num_classes)

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet32(num_classes=10, stage_channels=None):
    return ResNetCifar(BasicBlock, [5, 5, 5], num_classes=num_classes, stage_channels=stage_channels)


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def build_test_loader(data_root, batch_size=128, num_workers=0):
    test_transform = get_test_transform()
    test_dir = os.path.join(data_root, "test")
    train_dir = os.path.join(data_root, "train")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    else:
        test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=test_transform, download=False)
    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return test_loader, test_dataset.classes


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model):
    try:
        from thop import profile
        import copy
        dummy = torch.randn(1, 3, 32, 32).to(next(model.parameters()).device)
        flops, _ = profile(copy.deepcopy(model), inputs=(dummy,), verbose=False)
        return flops
    except ImportError:
        pass

    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            cout, cin = m.out_channels, m.in_channels
            kh, kw = m.kernel_size
            h, w = 32 // m.stride[0], 32 // m.stride[1]
            flops += 2 * cin * kh * kw * cout * h * w
        elif isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features
    return flops


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += images.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def infer_channels_from_state_dict(state_dict):
    ch0 = state_dict["conv1.weight"].shape[0]
    ch1 = state_dict["layer2.0.conv1.weight"].shape[0]
    ch2 = state_dict["layer3.0.conv1.weight"].shape[0]
    return [ch0, ch1, ch2]


def load_ga_channels():
    with open(GA_RESULT_TXT, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("channels:"):
                channels_str = line.split(":", 1)[1].strip()
                return ast.literal_eval(channels_str)
    return None


def load_pso_runtime():
    if not os.path.exists(PSO_RESULT_TXT):
        return None
    with open(PSO_RESULT_TXT, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"总时间:\s*([\d.]+)s", content)
    return float(m.group(1)) if m else None


def load_ga_runtime():
    if not os.path.exists(GA_RESULT_TXT):
        return None
    with open(GA_RESULT_TXT, "r", encoding="utf-8") as f:
        content = f.read()
    m1 = re.search(r"ga_runtime_s:\s*([\d.]+)", content)
    m2 = re.search(r"kd_runtime_s:\s*([\d.]+)", content)
    ga_time = float(m1.group(1)) if m1 else 0.0
    kd_time = float(m2.group(1)) if m2 else 0.0
    return ga_time + kd_time


def save_show_close(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print("  已保存: {}".format(save_path))
    plt.close(fig)


def plot_comparison_bar(labels, values, title, ylabel, save_name, color_map=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = color_map if color_map else ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="black", width=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(values) * 0.01,
                "{:.2f}".format(val), ha="center", va="bottom", fontsize=11)
    save_path = os.path.join(RESULT_DIR, save_name)
    save_show_close(fig, save_path)


def plot_grouped_bar(categories, group_values, group_labels, title, ylabel, save_name):
    n_groups = len(categories)
    n_bars = len(group_values)
    bar_width = 0.8 / n_bars
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(n_groups)
    for i, (vals, label) in enumerate(zip(group_values, group_labels)):
        bars = ax.bar(indices + i * bar_width, vals, bar_width,
                      label=label, color=colors[i % len(colors)], edgecolor="black")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(max(v) for v in group_values) * 0.01,
                    "{:.2f}".format(val), ha="center", va="bottom", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(indices + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    save_path = os.path.join(RESULT_DIR, save_name)
    save_show_close(fig, save_path)


def plot_params_flops_grouped(models_info, save_name):
    names = [m["name"] for m in models_info]
    params_m = [m["params"] / 1e6 for m in models_info]
    flops_m = [m["flops"] / 1e6 for m in models_info]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    bars1 = ax1.bar(names, params_m, color=colors, edgecolor="black", width=0.5)
    ax1.set_title("参数量对比 (M)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("参数量 (M)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, params_m):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(params_m) * 0.01,
                 "{:.3f}M".format(val), ha="center", va="bottom", fontsize=10)

    bars2 = ax2.bar(names, flops_m, color=colors, edgecolor="black", width=0.5)
    ax2.set_title("FLOPs 对比 (M)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("FLOPs (M)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, flops_m):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(flops_m) * 0.01,
                 "{:.1f}M".format(val), ha="center", va="bottom", fontsize=10)

    fig.suptitle("各模型参数量与FLOPs对比", fontsize=15, fontweight="bold")
    save_path = os.path.join(RESULT_DIR, save_name)
    save_show_close(fig, save_path)


def plot_summary_table(models_info, save_name):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    headers = ["指标", "Baseline\nResNet32", "PSO\n剪枝", "GA\n剪枝"]
    rows = [
        ["通道配置", "[16, 32, 64]", str(models_info[1]["channels"]), str(models_info[2]["channels"])],
        ["精度 (%)", "{:.2f}".format(models_info[0]["acc"]),
         "{:.2f}".format(models_info[1]["acc"]), "{:.2f}".format(models_info[2]["acc"])],
        ["参数量", "{:,}".format(models_info[0]["params"]),
         "{:,}".format(models_info[1]["params"]), "{:,}".format(models_info[2]["params"])],
        ["FLOPs", "{:,.0f}".format(models_info[0]["flops"]),
         "{:,.0f}".format(models_info[1]["flops"]), "{:,.0f}".format(models_info[2]["flops"])],
        ["参数压缩率 (%)", "-",
         "{:.2f}".format(models_info[1]["comp_rate"]), "{:.2f}".format(models_info[2]["comp_rate"])],
        ["FLOPs下降率 (%)", "-",
         "{:.2f}".format(models_info[1]["flops_red"]), "{:.2f}".format(models_info[2]["flops_red"])],
        ["推理时间 (s)", "{:.2f}".format(models_info[0]["infer_time"]),
         "{:.2f}".format(models_info[1]["infer_time"]), "{:.2f}".format(models_info[2]["infer_time"])],
    ]

    col_widths = [0.15, 0.28, 0.28, 0.28]

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=col_widths)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if key[0] == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
            cell.set_edgecolor("#2F5496")
        elif key[1] == 0:
            cell.set_facecolor("#D6E4F0")
            cell.set_text_props(fontweight="bold", fontsize=10)
        elif key[0] % 2 == 1:
            cell.set_facecolor("#F2F2F2")

    ax.set_title("ResNet32 模型压缩方案对比汇总", fontsize=15, fontweight="bold", pad=20)
    save_path = os.path.join(RESULT_DIR, save_name)
    save_show_close(fig, save_path)


def plot_comp_flops_comparison(baseline_params, baseline_flops, models_info, save_name):
    comp_rates = [m["comp_rate"] for m in models_info]
    flops_reds = [m["flops_red"] for m in models_info]
    names = [m["name"] for m in models_info]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#DD8452", "#55A868"]

    bars1 = ax1.bar(names, comp_rates, color=colors, edgecolor="black", width=0.5)
    ax1.set_title("参数压缩率 (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("压缩率 (%)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(comp_rates) * 1.2)
    for bar, val in zip(bars1, comp_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                 "{:.2f}%".format(val), ha="center", va="bottom", fontsize=11)

    bars2 = ax2.bar(names, flops_reds, color=colors, edgecolor="black", width=0.5)
    ax2.set_title("FLOPs 下降率 (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("下降率 (%)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(flops_reds) * 1.2)
    for bar, val in zip(bars2, flops_reds):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                 "{:.2f}%".format(val), ha="center", va="bottom", fontsize=11)

    fig.suptitle("PSO与GA剪枝方案压缩效果对比", fontsize=15, fontweight="bold")
    save_path = os.path.join(RESULT_DIR, save_name)
    save_show_close(fig, save_path)


def plot_accuracy_comp_tradeoff(models_info, save_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    for m in models_info:
        color = "#4C72B0" if m["name"] == "Baseline" else ("#DD8452" if m["name"] == "PSO" else "#55A868")
        marker = "s" if m["name"] == "Baseline" else ("o" if m["name"] == "PSO" else "^")
        size = 200 if m["name"] == "Baseline" else 150
        ax.scatter(m["comp_rate"], m["acc"], c=color, s=size, marker=marker,
                   edgecolors="black", linewidths=1.5, zorder=5)
        offset_x = -1.5 if m["name"] != "GA" else 1.0
        offset_y = 0.3
        ax.annotate(m["name"], (m["comp_rate"], m["acc"]),
                    textcoords="offset points", xytext=(offset_x * 8, offset_y * 8),
                    fontsize=11, fontweight="bold")

    ax.set_xlabel("参数压缩率 (%)", fontsize=12)
    ax.set_ylabel("测试精度 (%)", fontsize=12)
    ax.set_title("精度-压缩率权衡 (Accuracy vs Compression)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    save_path = os.path.join(RESULT_DIR, save_name)
    save_show_close(fig, save_path)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 70)
    print("     ResNet32 模型压缩方案对比分析")
    print("=" * 70)
    print("设备: {}".format(DEVICE))
    print("数据集: {}".format(DATA_ROOT))
    print("结果输出: {}".format(RESULT_DIR))

    print("\n[1/6] 加载CIFAR-10测试集...")
    test_loader, class_names = build_test_loader(DATA_ROOT)
    print("  测试样本数: {}".format(len(test_loader.dataset)))
    print("  类别: {}".format(class_names))

    print("\n[2/6] 加载三个模型...")

    print("  [Baseline] 加载 ResNet32 原始模型...")
    baseline_model = resnet32(num_classes=10).to(DEVICE)
    baseline_model.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
    baseline_model.eval()
    print("    通道: [16, 32, 64]")

    print("  [PSO] 加载 PSO剪枝模型...")
    pso_sd = torch.load(PSO_PATH, map_location=DEVICE)
    pso_channels = infer_channels_from_state_dict(pso_sd)
    pso_model = resnet32(num_classes=10, stage_channels=pso_channels).to(DEVICE)
    pso_model.load_state_dict(pso_sd)
    pso_model.eval()
    print("    通道: {}".format(pso_channels))

    print("  [GA] 加载 GA剪枝模型...")
    ga_channels = load_ga_channels()
    if ga_channels is None:
        ga_sd = torch.load(GA_PATH, map_location=DEVICE)
        ga_channels = infer_channels_from_state_dict(ga_sd)
        ga_model = resnet32(num_classes=10, stage_channels=ga_channels).to(DEVICE)
        ga_model.load_state_dict(ga_sd)
    else:
        ga_model = resnet32(num_classes=10, stage_channels=ga_channels).to(DEVICE)
        ga_model.load_state_dict(torch.load(GA_PATH, map_location=DEVICE))
    ga_model.eval()
    print("    通道: {}".format(ga_channels))

    print("\n[3/6] 计算模型指标 (参数量 / FLOPs)...")
    baseline_params = count_params(baseline_model)
    baseline_flops = estimate_flops(baseline_model)

    pso_params = count_params(pso_model)
    pso_flops = estimate_flops(pso_model)

    ga_params = count_params(ga_model)
    ga_flops = estimate_flops(ga_model)

    pso_comp_rate = (1 - pso_params / baseline_params) * 100
    pso_flops_red = (1 - pso_flops / baseline_flops) * 100 if baseline_flops > 0 else 0.0

    ga_comp_rate = (1 - ga_params / baseline_params) * 100
    ga_flops_red = (1 - ga_flops / baseline_flops) * 100 if baseline_flops > 0 else 0.0

    print("  Baseline: 参数={:,}  FLOPs={:,.0f}".format(baseline_params, baseline_flops))
    print("  PSO:      参数={:,}  压缩率={:.2f}%  FLOPs={:,.0f}  下降率={:.2f}%".format(
        pso_params, pso_comp_rate, pso_flops, pso_flops_red))
    print("  GA:       参数={:,}  压缩率={:.2f}%  FLOPs={:,.0f}  下降率={:.2f}%".format(
        ga_params, ga_comp_rate, ga_flops, ga_flops_red))

    print("\n[4/6] 评估测试精度与推理时间...")

    torch.cuda.synchronize()
    t0 = time.time()
    baseline_acc = evaluate(baseline_model, test_loader)
    torch.cuda.synchronize()
    baseline_infer = time.time() - t0

    torch.cuda.synchronize()
    t0 = time.time()
    pso_acc = evaluate(pso_model, test_loader)
    torch.cuda.synchronize()
    pso_infer = time.time() - t0

    torch.cuda.synchronize()
    t0 = time.time()
    ga_acc = evaluate(ga_model, test_loader)
    torch.cuda.synchronize()
    ga_infer = time.time() - t0

    print("  Baseline: 精度={:.2f}%  推理时间={:.3f}s".format(baseline_acc, baseline_infer))
    print("  PSO:      精度={:.2f}%  推理时间={:.3f}s".format(pso_acc, pso_infer))
    print("  GA:       精度={:.2f}%  推理时间={:.3f}s".format(ga_acc, ga_infer))

    print("\n[5/6] 读取算法运行时间...")
    pso_runtime = load_pso_runtime()
    ga_runtime = load_ga_runtime()

    if pso_runtime:
        print("  PSO 算法运行时间: {:.2f}s ({:.1f}min)".format(pso_runtime, pso_runtime / 60))
    else:
        print("  PSO 算法运行时间: 无法获取")
    if ga_runtime:
        print("  GA  算法运行时间: {:.2f}s ({:.1f}min)".format(ga_runtime, ga_runtime / 60))
    else:
        print("  GA  算法运行时间: 无法获取")

    baseline_model_info = {
        "name": "Baseline", "channels": [16, 32, 64],
        "params": baseline_params, "flops": baseline_flops,
        "acc": baseline_acc, "infer_time": baseline_infer,
        "comp_rate": 0.0, "flops_red": 0.0, "runtime": None,
    }
    pso_model_info = {
        "name": "PSO", "channels": pso_channels,
        "params": pso_params, "flops": pso_flops,
        "acc": pso_acc, "infer_time": pso_infer,
        "comp_rate": pso_comp_rate, "flops_red": pso_flops_red,
        "runtime": pso_runtime,
    }
    ga_model_info = {
        "name": "GA", "channels": ga_channels,
        "params": ga_params, "flops": ga_flops,
        "acc": ga_acc, "infer_time": ga_infer,
        "comp_rate": ga_comp_rate, "flops_red": ga_flops_red,
        "runtime": ga_runtime,
    }
    all_info = [baseline_model_info, pso_model_info, ga_model_info]

    print("\n" + "=" * 70)
    print("                    最终对比结果汇总")
    print("=" * 70)
    print("{:<18} {:>12} {:>12} {:>12}".format("指标", "Baseline", "PSO", "GA"))
    print("-" * 70)
    print("{:<18} {:>12} {:>12} {:>12}".format(
        "通道配置", "[16,32,64]", str(pso_channels), str(ga_channels)))
    print("{:<18} {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
        "测试精度", baseline_acc, pso_acc, ga_acc))
    print("{:<18} {:>12,} {:>12,} {:>12,}".format(
        "参数量", baseline_params, pso_params, ga_params))
    print("{:<18} {:>11.1f}M {:>11.1f}M {:>11.1f}M".format(
        "FLOPs", baseline_flops / 1e6, pso_flops / 1e6, ga_flops / 1e6))
    print("{:<18} {:>12} {:>11.2f}% {:>11.2f}%".format(
        "参数压缩率", "-", pso_comp_rate, ga_comp_rate))
    print("{:<18} {:>12} {:>11.2f}% {:>11.2f}%".format(
        "FLOPs下降率", "-", pso_flops_red, ga_flops_red))
    print("{:<18} {:>11.3f}s {:>11.3f}s {:>11.3f}s".format(
        "推理时间", baseline_infer, pso_infer, ga_infer))
    if pso_runtime and ga_runtime:
        print("{:<18} {:>12} {:>11.1f}s {:>11.1f}s".format(
            "算法运行时间", "-", pso_runtime, ga_runtime))
    print("{:<18} {:>11.4f}M {:>11.4f}M {:>11.4f}M".format(
        "参数量(M)", baseline_params / 1e6, pso_params / 1e6, ga_params / 1e6))
    print("=" * 70)

    print("\n[6/6] 生成可视化结果...")

    plot_summary_table(all_info, "summary_table.png")

    plot_params_flops_grouped(all_info, "params_flops_comparison.png")

    pruned_info = [pso_model_info, ga_model_info]
    plot_comp_flops_comparison(baseline_params, baseline_flops, pruned_info,
                               "comp_flops_comparison.png")

    plot_accuracy_comp_tradeoff(all_info, "accuracy_comp_tradeoff.png")

    all_names = [m["name"] for m in all_info]
    all_accs = [m["acc"] for m in all_info]
    plot_comparison_bar(all_names, all_accs,
                        "各模型测试精度对比", "精度 (%)", "accuracy_comparison.png")

    pruned_names = [m["name"] for m in pruned_info]
    comp_rates = [m["comp_rate"] for m in pruned_info]
    plot_comparison_bar(pruned_names, comp_rates,
                        "剪枝方案参数压缩率对比", "压缩率 (%)", "comp_rate_comparison.png")

    flops_rates = [m["flops_red"] for m in pruned_info]
    plot_comparison_bar(pruned_names, flops_rates,
                        "剪枝方案FLOPs下降率对比", "下降率 (%)", "flops_red_comparison.png")

    infer_times = [m["infer_time"] for m in all_info]
    plot_comparison_bar(all_names, infer_times,
                        "各模型推理时间对比", "时间 (s)", "infer_time_comparison.png")

    print("\n" + "=" * 70)
    print("对比分析完成! 所有结果已输出到:")
    print("  {}".format(RESULT_DIR))
    print("=" * 70)


if __name__ == "__main__":
    main()