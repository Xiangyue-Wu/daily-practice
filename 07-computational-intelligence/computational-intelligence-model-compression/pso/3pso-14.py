"""
PSO Iteration 14 - ResNet32 迭代剪枝 (从pso-13最优模型[7,10,30]继续迭代剪枝)
==========================================================
策略: 迭代剪枝 - 加载pso-13最优模型[7,10,30]，在已微调权重上计算L1重要性
  测试少数激进候选([6,10,30]/[7,9,30]/[6,9,30]/[7,10,29])
  使用80epoch微调提供更多恢复时间

pso-13结果: ch=[7,10,30] acc=89.74% comp=80.59%
目标: 在[7,10,30]基础上进一步压缩
方法: 迭代剪枝 - 从已微调模型出发再次计算L1重要性并剪枝
"""

import os, time, copy, random, argparse
from collections import OrderedDict
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

TRAIN_RATIO = 1.0
TEST_RATIO = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"data/cifar10"

# ============================================================
# 修改模型路径：改下面三行即可
# ============================================================
PRETRAINED_PATH = r"models/resnet32_cifar10_best.pth"
PSO13_MODEL_PATH = r"models/resnet32_pso_pruned.pth"
PSO_SAVE_PATH = r"models/resnet32_pso_pruned.pth"
PSO_RESULT_DIR = r"outputs/re-pso-14"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def conv3x3(in_plane, out_plane, stride=1):
    return nn.Conv2d(in_plane, out_plane, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
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
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; return self.relu(out)


class ResNetCifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, norm_layer=nn.BatchNorm2d,
                 stage_channels=None):
        super().__init__()
        if stage_channels is None: stage_channels = [16, 32, 64]
        self.stage_channels = stage_channels; self.in_planes = stage_channels[0]
        self.conv1 = conv3x3(3, stage_channels[0])
        self.bn1 = norm_layer(stage_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, stage_channels[0], layers[0], 1, norm_layer)
        self.layer2 = self._make_layer(block, stage_channels[1], layers[1], 2, norm_layer)
        self.layer3 = self._make_layer(block, stage_channels[2], layers[2], 2, norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_channels[2] * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, num_blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion))
        layers_list = [block(self.in_planes, planes, stride, downsample, norm_layer)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers_list.append(block(self.in_planes, planes, 1, None, norm_layer))
        return nn.Sequential(*layers_list)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); return self.fc(x.view(x.size(0), -1))


def resnet32(num_classes=10, stage_channels=None):
    return ResNetCifar(BasicBlock, [5, 5, 5], num_classes=num_classes,
                       stage_channels=stage_channels)


def get_cifar10_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    return train_transform, test_transform


def build_dataloader(data_root, batch_size, num_workers, download,
                     train_ratio=1.0, test_ratio=1.0):
    train_transform, test_transform = get_cifar10_transforms()
    dataset_type = "imagefolder" if os.path.isdir(os.path.join(data_root, "train")) else "cifar10"
    if dataset_type == "imagefolder":
        train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"),
                                              transform=train_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_root, "test"),
                                             transform=test_transform)
    else:
        train_dataset = datasets.CIFAR10(data_root, train=True, transform=train_transform,
                                          download=download)
        test_dataset = datasets.CIFAR10(data_root, train=False, transform=test_transform,
                                         download=download)
    pin_memory = torch.cuda.is_available()
    train_n = int(len(train_dataset) * train_ratio)
    train_idx = random.sample(range(len(train_dataset)), train_n)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_n = int(len(test_dataset) * test_ratio)
    test_idx = random.sample(range(len(test_dataset)), test_n)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader, None, test_dataset


def evaluate(model, loader, criterion, device):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += images.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 32, 32).to(next(model.parameters()).device)
        flops, _ = profile(copy.deepcopy(model), inputs=(dummy,), verbose=False)
        return flops
    except ImportError:
        flops = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                cout, cin, kh, kw = m.out_channels, m.in_channels, *m.kernel_size
                h, w = 32 // m.stride[0], 32 // m.stride[1]
                flops += 2 * cin * kh * kw * cout * h * w
            elif isinstance(m, nn.Linear):
                flops += 2 * m.in_features * m.out_features
        return flops


def infer_channels_from_state_dict(state_dict):
    ch0 = state_dict['conv1.weight'].shape[0]
    ch1 = state_dict['layer2.0.conv1.weight'].shape[0]
    ch2 = state_dict['layer3.0.conv1.weight'].shape[0]
    return [ch0, ch1, ch2]


def compute_l1_importance(model):
    l1_map = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.abs()
            l1_map[name] = w.view(w.size(0), -1).sum(dim=1).cpu()
    return l1_map


def build_pruned_from_source(source_model, new_channels, source_channels):
    l1_map = compute_l1_importance(source_model)
    kept_out = {}
    for name, l1 in l1_map.items():
        if 'layer2' in name: stage = 2
        elif 'layer3' in name: stage = 3
        else: stage = 1
        if stage == 1 and name == 'conv1': nc = min(new_channels[0], l1.size(0))
        elif stage <= 3: nc = min(new_channels[stage - 1], l1.size(0))
        else: continue
        _, top_idx = torch.sort(l1, descending=True)
        kept_out[name] = sorted(top_idx[:nc].tolist())
    kept_in = {}
    for name in kept_out:
        if name == 'conv1': kept_in[name] = list(range(3)); continue
        if 'layer' not in name: continue
        parts = name.split('.')
        try: block_idx = int(parts[1])
        except ValueError:
            stage = 2 if 'layer2' in name else 3
            kept_in[name] = list(range(min(new_channels[stage - 1], source_channels[stage - 1]))); continue
        conv_type = parts[2] if len(parts) > 2 else ''
        if conv_type == 'conv1':
            if block_idx == 0:
                prev_conv = 'layer1.4.conv2' if 'layer2' in name else ('layer2.4.conv2' if 'layer3' in name else 'conv1')
                kept_in[name] = kept_out.get(prev_conv, list(range(min(source_channels[0], new_channels[0]))))
            else:
                stage = 2 if 'layer2' in name else (3 if 'layer3' in name else 1)
                kept_in[name] = kept_out.get(f'layer{stage}.{block_idx - 1}.conv2',
                                              list(range(min(new_channels[stage - 1], source_channels[stage - 1]))))
        elif conv_type == 'conv2':
            stage = 2 if 'layer2' in name else (3 if 'layer3' in name else 1)
            kept_in[name] = kept_out.get(f'layer{stage}.{block_idx}.conv1',
                                          list(range(min(new_channels[stage - 1], source_channels[stage - 1]))))
        elif conv_type == 'downsample':
            prev_conv = 'layer1.4.conv2' if 'layer2' in name else ('layer2.4.conv2' if 'layer3' in name else 'conv1')
            kept_in[name] = kept_out.get(prev_conv, list(range(min(16, source_channels[0]))))
        else:
            stage = 2 if 'layer2' in name else (3 if 'layer3' in name else 1)
            kept_in[name] = list(range(min(new_channels[stage - 1], source_channels[stage - 1])))
    pruned = resnet32(num_classes=10, stage_channels=new_channels); pruned.to(DEVICE)
    orig_sd, new_sd = source_model.state_dict(), OrderedDict()
    last_out_key = 'layer3.4.conv2'
    last_out_indices = kept_out.get(last_out_key, list(range(min(new_channels[2], source_channels[2]))))
    for name, target_tensor in pruned.state_dict().items():
        if name not in orig_sd: new_sd[name] = target_tensor.clone(); continue
        orig, new_shape, orig_shape = orig_sd[name], target_tensor.shape, orig_sd[name].shape
        if orig_shape == new_shape: new_sd[name] = orig.clone(); continue
        if 'weight' in name and len(orig_shape) == 4:
            conv_base = name.replace('.weight', '')
            out_idx = kept_out.get(conv_base, list(range(min(new_shape[0], orig_shape[0]))))
            in_idx = kept_in.get(conv_base, list(range(min(new_shape[1], orig_shape[1]))))
            new_w = torch.zeros(new_shape)
            for ni, oi in enumerate(out_idx[:new_shape[0]]):
                for nj, ij in enumerate(in_idx[:new_shape[1]]):
                    if oi < orig_shape[0] and ij < orig_shape[1]: new_w[ni, nj] = orig[oi, ij]
            new_sd[name] = new_w
        elif len(orig_shape) == 1:
            conv_base = name
            for ext in ['.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked']:
                conv_base = conv_base.replace(ext, '')
            conv_base = conv_base.replace('bn1', 'conv1').replace('bn2', 'conv2').replace('downsample.1', 'downsample.0')
            success = False
            if conv_base in kept_out:
                out_idx = kept_out[conv_base]; new_w = torch.zeros(new_shape)
                for ni, oi in enumerate(out_idx[:new_shape[0]]):
                    if oi < orig_shape[0]: new_w[ni] = orig[oi]
                new_sd[name] = new_w; success = True
            if not success:
                for rp in [('bn1', 'conv1'), ('bn2', 'conv2'), ('downsample.1', 'downsample.0')]:
                    candidate = conv_base.replace(rp[0], rp[1])
                    if candidate != conv_base and candidate in kept_out:
                        out_idx = kept_out[candidate]; new_w = torch.zeros(new_shape)
                        for ni, oi in enumerate(out_idx[:new_shape[0]]):
                            if oi < orig_shape[0]: new_w[ni] = orig[oi]
                        new_sd[name] = new_w; success = True; break
            if not success: new_sd[name] = orig[:min(new_shape[0], orig_shape[0])].clone()
        elif 'fc' in name:
            new_w = torch.zeros(new_shape); min_out = min(new_shape[0], orig_shape[0])
            for nj, ij in enumerate(last_out_indices[:new_shape[1]]):
                if ij < orig_shape[1]: new_w[:min_out, nj] = orig[:min_out, ij]
            new_sd[name] = new_w
        elif len(orig_shape) == 2:
            new_sd[name] = orig[:min(new_shape[0], orig_shape[0]), :min(new_shape[1], orig_shape[1])].clone()
        else: new_sd[name] = orig.clone()
    for n in pruned.state_dict().keys():
        if n not in new_sd: new_sd[n] = pruned.state_dict()[n].clone()
    pruned.load_state_dict(new_sd, strict=True)
    return pruned


def quick_train_eval(model, train_loader, test_loader, device, epochs, lr, wd):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(); loss = criterion(model(images), labels)
            loss.backward(); optimizer.step()
    return evaluate(model, test_loader, criterion, device)


def train_model(model, train_loader, test_loader, device, epochs, lr, wd, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)
    best_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for epoch in range(epochs):
        model.train(); total_loss, correct, total = 0.0, 0, 0; t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(images)
            loss = criterion(outputs, labels); loss.backward(); optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item(); total += images.size(0)
        scheduler.step()
        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        test_acc = evaluate(model, test_loader, criterion, device); dt = time.time() - t0
        if test_acc > best_acc: best_acc = test_acc; torch.save(model.state_dict(), save_path)
        print(f"  Epoch [{epoch + 1:03d}/{epochs:03d}] Loss={train_loss:.4f} "
              f"Train={train_acc:.2f}% Test={test_acc:.2f}% Best={best_acc:.2f}% Time={dt:.1f}s")
    return best_acc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--finetune_epochs", type=int, default=80)
    p.add_argument("--quick_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4, dest="wd")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed); total_t0 = time.time()
    print("=" * 65)
    print("PSO第14轮迭代 - 迭代剪枝 (从pso-13模型[7,10,30]继续)")
    print(f"设备: {DEVICE}")
    print("=" * 65)

    print("\n[1/7] 加载CIFAR-10...")
    train_loader, test_loader, _, _ = build_dataloader(
        DATA_ROOT, args.batch_size, args.num_workers, download=False,
        train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO)
    print(f"  训练: {len(train_loader.dataset)}, 测试: {len(test_loader.dataset)}")

    if os.path.exists(PSO_SAVE_PATH):
        print(f"\n[跳过训练] 检测到已存在最优模型: {PSO_SAVE_PATH}")
        print("  直接加载进行评估...")
        criterion = nn.CrossEntropyLoss()

        state_dict = torch.load(PSO_SAVE_PATH, map_location=DEVICE)
        best_ch = infer_channels_from_state_dict(state_dict)
        print(f"  推断通道配置: {best_ch}")
        pruned_model = resnet32(num_classes=10, stage_channels=best_ch).to(DEVICE)
        pruned_model.load_state_dict(state_dict)
        pruned_acc = evaluate(pruned_model, test_loader, criterion, DEVICE)
        pruned_params = count_params(pruned_model)
        pruned_flops = estimate_flops(pruned_model)

        has_pretrained = os.path.exists(PRETRAINED_PATH)
        if has_pretrained:
            original_model = resnet32(num_classes=10).to(DEVICE)
            original_model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
            original_acc = evaluate(original_model, test_loader, criterion, DEVICE)
            original_params = count_params(original_model)
            original_flops = estimate_flops(original_model)
            param_comp_pct = (1 - pruned_params / original_params) * 100
            flops_red_pct = (1 - pruned_flops / original_flops) * 100 if original_flops > 0 else 0.0
        else:
            original_acc = None
            original_params = None
            original_flops = None
            original_params_ref = 470218
            param_comp_pct = (1 - pruned_params / original_params_ref) * 100
            flops_red_pct = None

        pso13_acc = None; pso13_comp = None
        if os.path.exists(PSO13_MODEL_PATH):
            pso13_model = resnet32(num_classes=10, stage_channels=[7, 10, 30]).to(DEVICE)
            pso13_model.load_state_dict(torch.load(PSO13_MODEL_PATH, map_location=DEVICE))
            pso13_acc = evaluate(pso13_model, test_loader, criterion, DEVICE)
            pso13_params = count_params(pso13_model)
            if has_pretrained:
                pso13_comp = (1 - pso13_params / original_params) * 100
            else:
                pso13_comp = (1 - pso13_params / original_params_ref) * 100

        print("\n" + "=" * 65)
        print("                    pso-14 模型评估结果 (直接加载)")
        print("=" * 65)
        print(f"  通道配置: {best_ch}")
        print(f"  模型精度: {pruned_acc:.2f}%")
        print(f"  参数压缩率: {param_comp_pct:.2f}%")
        if flops_red_pct is not None:
            print(f"  FLOPs下降率: {flops_red_pct:.2f}%")
        if pso13_acc is not None:
            print(f"  pso-13基准: ch=[7,10,30] acc={pso13_acc:.2f}% comp={pso13_comp:.2f}%")
        if original_params is not None:
            print(f"  原始参数: {original_params:,} -> 剪枝后: {pruned_params:,}")
        else:
            print(f"  剪枝后参数: {pruned_params:,}")
        if original_flops is not None:
            print(f"  原始FLOPs: {original_flops:,.0f} -> 剪枝后: {pruned_flops:,.0f}")
        print(f"  原始ResNet32参数参考值: {original_params_ref:,}")
        print("=" * 65)

        result_path = os.path.join(PSO_RESULT_DIR, "pso_result.txt")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                existing_result = f.read()
            print(f"\n历史结果 ({result_path}):")
            print(existing_result)
        return

    print("\n[2/7] 加载预训练 & pso-13最优模型...")
    original_model = resnet32(num_classes=10).to(DEVICE)
    original_model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))

    pso13_model = resnet32(num_classes=10, stage_channels=[7, 10, 30]).to(DEVICE)
    pso13_model.load_state_dict(torch.load(PSO13_MODEL_PATH, map_location=DEVICE))
    print(f"  pso-13模型已加载: [7,10,30]")

    print("\n[3/7] 评估原始...")
    criterion = nn.CrossEntropyLoss()
    original_acc = evaluate(original_model, test_loader, criterion, DEVICE)
    original_params = count_params(original_model)
    original_flops = estimate_flops(original_model)
    pso13_acc = evaluate(pso13_model, test_loader, criterion, DEVICE)
    pso13_params = count_params(pso13_model)
    print(f"  原始精度: {original_acc:.2f}% 参数: {original_params:,}")
    print(f"  pso-13精度: {pso13_acc:.2f}% 参数: {pso13_params:,}")

    print("\n[4/7] 迭代剪枝候选评估...")
    source_channels = [7, 10, 30]
    candidates = [
        [6, 10, 30],
        [7, 9, 30],
        [6, 9, 30],
        [7, 10, 29],
    ]
    print("=" * 65)
    print(f"迭代剪枝 from [7,10,30] (基于已微调L1重要性)")
    print("=" * 65)
    results = []
    for ch in candidates:
        pruned = build_pruned_from_source(pso13_model, ch, source_channels); pruned.to(DEVICE)
        params = count_params(pruned); comp = (1 - params / original_params) * 100
        t0 = time.time()
        acc = quick_train_eval(pruned, train_loader, test_loader, DEVICE, args.quick_epochs, args.lr, args.wd)
        dt = time.time() - t0
        results.append((ch, comp, acc, dt))
        print(f"  ch={ch} comp={comp:.1f}% q_acc={acc:.2f}% ({dt:.1f}s)")
    results.sort(key=lambda x: (x[2], x[1]), reverse=True)
    print(f"\n  Top候选:")
    for i, (ch, comp, acc, dt) in enumerate(results):
        print(f"    {i + 1}. ch={ch} comp={comp:.1f}% q_acc={acc:.2f}%")
    best_ch, best_comp, best_acc, _ = results[0]
    print(f"\n  选择: ch={best_ch} comp={best_comp:.1f}% q_acc={best_acc:.2f}%")
    print("=" * 65)

    print(f"\n[5/7] 微调 ({args.finetune_epochs} epochs, 迭代剪枝需更多恢复)...")
    pruned_model = build_pruned_from_source(pso13_model, best_ch, source_channels); pruned_model.to(DEVICE)
    print(f"  剪枝后参数: {count_params(pruned_model):,}")
    ft_t0 = time.time()
    train_model(pruned_model, train_loader, test_loader, DEVICE,
                args.finetune_epochs, args.lr, args.wd, PSO_SAVE_PATH)
    ft_time = time.time() - ft_t0

    print("\n[6/7] 最终评估...")
    if os.path.exists(PSO_SAVE_PATH):
        pruned_model.load_state_dict(torch.load(PSO_SAVE_PATH, map_location=DEVICE))
    pruned_acc = evaluate(pruned_model, test_loader, criterion, DEVICE)
    pruned_params = count_params(pruned_model); pruned_flops = estimate_flops(pruned_model)
    total_time = time.time() - total_t0
    param_comp_pct = (1 - pruned_params / original_params) * 100
    flops_red_pct = (1 - pruned_flops / original_flops) * 100 if original_flops > 0 else 0.0

    print("\n[7/7] 结果对比...")
    pso13_comp = (1 - pso13_params / original_params) * 100
    print(f"  pso-13: ch=[7,10,30] acc={pso13_acc:.2f}% comp={pso13_comp:.2f}%")
    print(f"  pso-14: ch={best_ch} acc={pruned_acc:.2f}% comp={param_comp_pct:.2f}%")
    if pruned_acc >= 88.0:
        print(f"  ✓ 达标! 压缩率{'提升' if param_comp_pct > pso13_comp else '未提升'}: "
              f"{param_comp_pct - pso13_comp:+.2f}%")
    else:
        print(f"  ✗ 未达标 (需≥88%)")

    print("\n" + "=" * 65)
    print("                    第14轮压缩结果汇总")
    print("=" * 65)
    print(f"  a) 算法运行时间:        {total_time:.2f}s")
    print(f"     - 候选评估时间:       {sum(r[3] for r in results):.2f}s")
    print(f"     - 微调训练时间:       {ft_time:.2f}s")
    print(f"  b) 模型参数压缩率:       {param_comp_pct:.2f}%")
    print(f"  c) FLOPs下降率:          {flops_red_pct:.2f}%")
    print(f"  d) 压缩前精度:           {original_acc:.2f}%")
    print(f"     压缩后精度:           {pruned_acc:.2f}%")
    print(f"  -------------------------------------------------")
    print(f"  原始通道: [16, 32, 64]")
    print(f"  剪枝后通道: {best_ch}")
    print(f"  原始参数: {original_params:,} -> 剪枝后: {pruned_params:,}")
    print(f"  原始FLOPs: {original_flops:,.0f} -> 剪枝后: {pruned_flops:,.0f}")
    print("=" * 65)

    os.makedirs(PSO_RESULT_DIR, exist_ok=True)
    with open(os.path.join(PSO_RESULT_DIR, "pso_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"迭代: 14\n策略: 迭代剪枝 from pso-13模型[7,10,30]\n")
        f.write(f"总时间: {total_time:.2f}s\n微调: {ft_time:.2f}s\n")
        f.write(f"压缩率: {param_comp_pct:.2f}%\nFLOPs: {flops_red_pct:.2f}%\n")
        f.write(f"精度: {original_acc:.2f}% -> {pruned_acc:.2f}%\n")
        f.write(f"通道: {best_ch}\n参数: {pruned_params}\n")
    print(f"\n结果保存至: {PSO_RESULT_DIR}/pso_result.txt")

if __name__ == "__main__":
    main()