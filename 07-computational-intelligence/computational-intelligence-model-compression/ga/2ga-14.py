import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = r"."
DATA_ROOT = os.path.join(BASE_DIR, "cifar10")
PRETRAINED_PATH = os.path.join(BASE_DIR, "resnet32_cifar10_best.pth")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "re-ga-14")
SAVE_PATH = os.path.join(RESULT_DIR, "ga-pruned_best.pth")

TRAIN_RATIO = 1.0
TEST_RATIO = 1.0

os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    def __init__(self, block, layers, num_classes=10, norm_layer=nn.BatchNorm2d,
                 stage_channels=None):
        super(ResNetCifar, self).__init__()
        if stage_channels is None:
            stage_channels = [16, 32, 64]
        self.in_planes = stage_channels[0]
        self.stage_channels = stage_channels

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


def resnet32(num_classes=10, stage_channels=None):
    return ResNetCifar(BasicBlock, [5, 5, 5], num_classes=num_classes, stage_channels=stage_channels)


def get_cifar10_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return train_transform, test_transform


def detect_dataset_type(data_root):
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        return "imagefolder"
    return "cifar10"


def build_dataloader(data_root, batch_size, num_workers, download, train_ratio=1.0, test_ratio=1.0):
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
    random.shuffle(test_indices)
    test_subset = Subset(test_dataset, test_indices[:test_subset_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, test_loader, class_names


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
    avg_loss = total_loss / total_num if total_num > 0 else 0.0
    avg_acc = 100.0 * total_correct / total_num if total_num > 0 else 0.0
    return avg_loss, avg_acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model):
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            cout = m.out_channels
            cin = m.in_channels
            flops += 2 * cin * kh * kw * cout * 32 * 32
        elif isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features
    return flops


def channels_from_ratios(keep_ratios):
    original_c = [16, 32, 64]
    new_channels = []
    for i in range(3):
        orig_c = original_c[i]
        keep_r = max(0.10, min(1.0, keep_ratios[i]))
        new_c = max(2, min(orig_c, int(orig_c * keep_r + 0.5)))
        new_channels.append(new_c)
    return new_channels


def compute_stage_importance(original_model):
    stage1_imp = torch.zeros(16)
    stage2_imp = torch.zeros(32)
    stage3_imp = torch.zeros(64)

    for name, module in original_model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data.cpu()
            imp = torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
            if "layer1" in name:
                stage1_imp += imp
            elif "layer2" in name:
                stage2_imp += imp
            elif "layer3" in name:
                stage3_imp += imp

    return [stage1_imp, stage2_imp, stage3_imp]


def compute_importance_ratio(keep_ratios, stage_importance):
    original_c = [16, 32, 64]
    ratios = []
    for i in range(3):
        new_c = channels_from_ratios(keep_ratios)[i]
        imp = stage_importance[i]
        total_imp = imp.sum()
        if new_c >= original_c[i]:
            kept_imp = total_imp
        else:
            _, idx = torch.topk(imp, new_c)
            kept_imp = imp[idx].sum()
        ratios.append((kept_imp / (total_imp + 1e-10)).item())
    return np.mean(ratios)


class KDLoss(nn.Module):
    def __init__(self, T=5.0, alpha=0.7):
        super(KDLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        hard_loss = F.cross_entropy(student_logits, targets)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T * self.T)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


def train_with_kd(student_model, teacher_model, train_loader, test_loader, device,
                  epochs, lr, weight_decay, save_path, T=5.0, alpha=0.7):
    kd_loss_fn = KDLoss(T=T, alpha=alpha)
    hard_criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(epochs * 0.5), int(epochs * 0.75)],
        gamma=0.1
    )

    teacher_model.eval()
    best_acc = 0.0

    for epoch in range(epochs):
        student_model.train()
        total_loss = 0.0
        total_correct = 0
        total_num = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher_model(images).detach()

            optimizer.zero_grad()
            student_logits = student_model(images)
            loss = kd_loss_fn(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = student_logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_num += images.size(0)

        scheduler.step()

        train_loss = total_loss / total_num if total_num > 0 else 0.0
        train_acc = 100.0 * total_correct / total_num if total_num > 0 else 0.0
        test_loss, test_acc = evaluate_model(student_model, test_loader, hard_criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student_model.state_dict(), save_path)

        print("  Epoch [{:03d}/{:03d}] Train Loss: {:.4f} Train Acc: {:.2f}% "
              "Test Loss: {:.4f} Test Acc: {:.2f}% Best: {:.2f}%".format(
                  epoch + 1, epochs, train_loss, train_acc, test_loss, test_acc, best_acc))

    return best_acc


class GAForChannelPruning:
    def __init__(self, pop_size=30, n_generations=30, crossover_prob=0.85,
                 mutation_prob=0.47, tournament_size=3,
                 original_params=None, stage_importance=None,
                 acc_weight=0.42, comp_weight=0.58):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.dim = 3
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.original_params = original_params
        self.stage_importance = stage_importance
        self.acc_weight = acc_weight
        self.comp_weight = comp_weight

        self.lb = np.array([0.09, 0.10, 0.20])
        self.ub = np.array([1.0, 0.62, 1.0])

        self.population = np.random.uniform(low=self.lb, high=self.ub, size=(pop_size, self.dim))

        self.fitness_cache = {}
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.best_compression = 0.0
        self.best_channels = [16, 32, 64]
        self.best_imp_ratio = 0.0

    def _individual_key(self, individual):
        return tuple(channels_from_ratios(individual))

    def evaluate_individual(self, keep_ratios):
        key = self._individual_key(keep_ratios)
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        new_channels = channels_from_ratios(keep_ratios)
        pruned_model = resnet32(num_classes=10, stage_channels=new_channels)
        pruned_params = count_parameters(pruned_model)
        del pruned_model

        compression_ratio = 1.0 - pruned_params / self.original_params if self.original_params > 0 else 0.0
        imp_ratio = compute_importance_ratio(keep_ratios, self.stage_importance) if self.stage_importance else 0.5

        fitness = compression_ratio * self.comp_weight + imp_ratio * self.acc_weight

        result = (fitness, compression_ratio, imp_ratio, new_channels, pruned_params)
        self.fitness_cache[key] = result
        return result

    def tournament_selection(self, fitness_values):
        candidates = random.sample(range(self.pop_size), self.tournament_size)
        return max(candidates, key=lambda i: fitness_values[i])

    def sbx_crossover(self, parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()
        eta = 20.0
        for i in range(self.dim):
            if random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    rand = random.random()
                    beta = 1.0 + 2.0 * (y1 - self.lb[i]) / (y2 - y1 + 1e-10)
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                    child1[i] = np.clip(c1, self.lb[i], self.ub[i])
                    child2[i] = np.clip(c2, self.lb[i], self.ub[i])
        return child1, child2

    def polynomial_mutation(self, individual):
        mutated = individual.copy()
        eta_m = 20.0
        for i in range(self.dim):
            if random.random() <= self.mutation_prob:
                rand = random.random()
                if rand <= 0.5:
                    delta = (2.0 * rand) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (eta_m + 1.0))
                mutated[i] = individual[i] + delta * (self.ub[i] - self.lb[i]) * 0.45
                mutated[i] = np.clip(mutated[i], self.lb[i], self.ub[i])
        return mutated

    def evolve(self, verbose=True):
        print("=" * 60)
        print("GA start (KD mode) - search pruning scheme")
        print("  Pop: {}, Gen: {}, Cross: {}, Mut: {}".format(
            self.pop_size, self.n_generations, self.crossover_prob, self.mutation_prob))
        print("  acc_weight: {}, comp_weight: {}".format(self.acc_weight, self.comp_weight))
        print("  lb: {}, ub: {}".format(self.lb.tolist(), self.ub.tolist()))
        print("=" * 60)

        fitness_values = np.zeros(self.pop_size)

        for generation in range(self.n_generations):
            gen_start = time.time()
            for i in range(self.pop_size):
                fitness, compression, imp_ratio, channels, params = self.evaluate_individual(self.population[i])
                fitness_values[i] = fitness
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = self.population[i].copy()
                    self.best_compression = compression
                    self.best_channels = channels
                    self.best_imp_ratio = imp_ratio

            gen_best = np.max(fitness_values)
            gen_avg = np.mean(fitness_values)
            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(gen_avg)

            if verbose:
                print("  Gen {:2d}/{:2d} | Best fit: {:.4f} | Avg: {:.4f} | Ch: {} | Comp: {:.2%} | Imp: {:.2%}".format(
                    generation + 1, self.n_generations, self.best_fitness, gen_avg,
                    self.best_channels, self.best_compression, self.best_imp_ratio))

            new_population = np.zeros_like(self.population)
            elite_size = max(1, self.pop_size // 10)
            elite_indices = np.argsort(fitness_values)[-elite_size:]
            for j, idx in enumerate(elite_indices):
                new_population[j] = self.population[idx].copy()

            for j in range(elite_size, self.pop_size, 2):
                p1 = self.tournament_selection(fitness_values)
                p2 = self.tournament_selection(fitness_values)
                if random.random() <= self.crossover_prob:
                    child1, child2 = self.sbx_crossover(self.population[p1], self.population[p2])
                else:
                    child1, child2 = self.population[p1].copy(), self.population[p2].copy()
                child1 = self.polynomial_mutation(child1)
                child2 = self.polynomial_mutation(child2)
                new_population[j] = child1
                if j + 1 < self.pop_size:
                    new_population[j + 1] = child2

            self.population = new_population
            if verbose:
                print("  Gen {:2d} done, time: {:.2f}s".format(generation + 1, time.time() - gen_start))

        print("=" * 60)
        print("GA complete! Best ch: {}, Comp: {:.2%}".format(self.best_channels, self.best_compression))
        print("=" * 60)
        return self.best_individual, self.best_fitness, self.best_compression, self.best_channels


def save_results(info, result_dir):
    path = os.path.join(result_dir, "results.txt")
    with open(path, "w") as f:
        for k, v in info.items():
            f.write("{}: {}\n".format(k, v))
    print("Results saved to: {}".format(path))


def load_pruned_model_from_results(save_path, result_dir, device):
    results_path = os.path.join(result_dir, "results.txt")
    channels = None
    with open(results_path, "r") as f:
        for line in f:
            if line.startswith("channels:"):
                channels_str = line.split(":", 1)[1].strip()
                channels = eval(channels_str)
                break
    if channels is None:
        raise ValueError("Could not find channels info in results.txt")

    model = resnet32(num_classes=10, stage_channels=channels).to(device)
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Loaded pruned model with channels: {}".format(channels))
    return model, channels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--n_generations", type=int, default=30)
    parser.add_argument("--crossover_prob", type=float, default=0.85)
    parser.add_argument("--mutation_prob", type=float, default=0.47)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--kd_T", type=float, default=5.0)
    parser.add_argument("--kd_alpha", type=float, default=0.7)
    parser.add_argument("--acc_weight", type=float, default=0.42)
    parser.add_argument("--comp_weight", type=float, default=0.58)
    parser.add_argument("--seed", type=int, default=88888)
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 60)
    print("GA-14: GA + KD Model Compression (L2 ub=0.62, KD T=5.0)")
    print("=" * 60)
    print("Device: {}, Result dir: {}".format(DEVICE, RESULT_DIR))
    print("Strategy: L2 ub=0.62 (max 19-20 ch boundary), KD T=5.0 for better distillation")
    print("  acc_weight={}, comp_weight={}, mutation_prob={}, n_generations={}, seed={}".format(
        args.acc_weight, args.comp_weight, args.mutation_prob, args.n_generations, args.seed))
    print("  lb=[0.09,0.10,0.20], ub=[1.0,0.62,1.0], KD T={}, alpha={}, epochs={}".format(
        args.kd_T, args.kd_alpha, args.finetune_epochs))

    if os.path.exists(SAVE_PATH):
        print("\n[Info] Best model found at: {}".format(SAVE_PATH))
        print("[Info] Loading model directly, skipping GA + KD training...\n")

        print("[Step 1] Loading CIFAR-10 test set...")
        _, test_loader, _ = build_dataloader(
            DATA_ROOT, args.batch_size, 0, download=False,
            train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO
        )

        print("\n[Step 2] Loading pruned model...")
        pruned_model, best_channels = load_pruned_model_from_results(SAVE_PATH, RESULT_DIR, DEVICE)

        print("\n[Step 3] Evaluating pruned model...")
        criterion = nn.CrossEntropyLoss()
        _, pruned_acc = evaluate_model(pruned_model, test_loader, criterion, DEVICE)
        pruned_params = count_parameters(pruned_model)
        pruned_flops = estimate_flops(pruned_model)

        original_params = 466906
        original_flops = 949847296

        param_comp = (1 - pruned_params / original_params) * 100
        flops_red = (1 - pruned_flops / original_flops) * 100

        print("\n  " + "=" * 50)
        print("  GA-14 KD Pruning Results (Loaded from disk)")
        print("  " + "=" * 50)
        print("  b) Param compression ratio: {:.2f}%".format(param_comp))
        print("  c) FLOPs reduction ratio:   {:.2f}%".format(flops_red))
        print("  d) Pruned accuracy:         {:.2f}%".format(pruned_acc))
        print("  Original channels: [16, 32, 64]")
        print("  Pruned channels:   {}".format(best_channels))
        print("  Original params:   {:,} ({:.4f}M)".format(original_params, original_params / 1e6))
        print("  Pruned params:     {:,} ({:.4f}M)".format(pruned_params, pruned_params / 1e6))
        print("  Original FLOPs:    {:,} ({:.2f}M)".format(original_flops, original_flops / 1e6))
        print("  Pruned FLOPs:      {:,} ({:.2f}M)".format(pruned_flops, pruned_flops / 1e6))
        print("  " + "=" * 50)

        print("\n" + "=" * 60)
        print("GA-14 complete!")
        print("=" * 60)
        return

    print("\n[Step 1] Loading CIFAR-10...")
    train_loader, test_loader, class_names = build_dataloader(
        DATA_ROOT, args.batch_size, 0, download=False,
        train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO
    )

    print("\n[Step 2] Loading pretrained ResNet32 (teacher)...")
    teacher_model = resnet32(num_classes=10).to(DEVICE)
    state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    teacher_model.load_state_dict(state_dict)
    teacher_model.eval()

    print("\n[Step 3] Evaluating teacher model...")
    criterion = nn.CrossEntropyLoss()
    _, original_acc = evaluate_model(teacher_model, test_loader, criterion, DEVICE)
    original_params = count_parameters(teacher_model)
    original_flops = estimate_flops(teacher_model)
    print("  Teacher accuracy: {:.2f}%".format(original_acc))
    print("  Teacher params: {:,} ({:.4f}M)".format(original_params, original_params / 1e6))
    print("  Teacher FLOPs: {:,} ({:.2f}M)".format(original_flops, original_flops / 1e6))

    print("\n[Step 4] Computing channel importance...")
    stage_importance = compute_stage_importance(teacher_model)

    print("\n[Step 5] GA search...")
    ga_start = time.time()
    ga = GAForChannelPruning(
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        original_params=original_params,
        stage_importance=stage_importance,
        acc_weight=args.acc_weight,
        comp_weight=args.comp_weight,
    )
    best_ratios, best_fitness, best_compression, best_channels = ga.evolve()
    ga_runtime = time.time() - ga_start

    print("\n[Step 6] Building pruned student model and KD training...")
    student_model = resnet32(num_classes=10, stage_channels=best_channels).to(DEVICE)
    student_params_before = count_parameters(student_model)
    print("  Student params: {:,} ({:.4f}M)".format(student_params_before, student_params_before / 1e6))
    print("  KD: T={}, alpha={}, epochs={}".format(args.kd_T, args.kd_alpha, args.finetune_epochs))

    ft_start = time.time()
    best_student_acc = train_with_kd(
        student_model, teacher_model, train_loader, test_loader, DEVICE,
        args.finetune_epochs, args.lr, args.weight_decay, SAVE_PATH,
        T=args.kd_T, alpha=args.kd_alpha
    )
    ft_runtime = time.time() - ft_start

    print("\n[Step 7] Final evaluation...")
    if os.path.exists(SAVE_PATH):
        student_model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        _, pruned_acc = evaluate_model(student_model, test_loader, criterion, DEVICE)
    else:
        pruned_acc = best_student_acc

    pruned_params = count_parameters(student_model)
    pruned_flops = estimate_flops(student_model)

    param_comp = (1 - pruned_params / original_params) * 100
    flops_red = (1 - pruned_flops / original_flops) * 100
    acc_loss = original_acc - pruned_acc
    total_runtime = ga_runtime + ft_runtime

    print("\n  " + "=" * 50)
    print("  GA-14 KD Pruning Results")
    print("  " + "=" * 50)
    print("  a) Algorithm runtime:       {:.2f}s (GA: {:.2f}s, KD: {:.2f}s)".format(total_runtime, ga_runtime, ft_runtime))
    print("  b) Param compression ratio: {:.2f}%".format(param_comp))
    print("  c) FLOPs reduction ratio:   {:.2f}%".format(flops_red))
    print("  d) Accuracy before:         {:.2f}%".format(original_acc))
    print("     Accuracy after:          {:.2f}%".format(pruned_acc))
    print("     Accuracy loss:           {:.2f}%".format(acc_loss))
    print("  Original channels: [16, 32, 64]")
    print("  Pruned channels:   {}".format(best_channels))
    print("  Original params:   {:,} ({:.4f}M)".format(original_params, original_params / 1e6))
    print("  Pruned params:     {:,} ({:.4f}M)".format(pruned_params, pruned_params / 1e6))
    print("  Original FLOPs:    {:,} ({:.2f}M)".format(original_flops, original_flops / 1e6))
    print("  Pruned FLOPs:      {:,} ({:.2f}M)".format(pruned_flops, pruned_flops / 1e6))
    print("  " + "=" * 50)

    results_info = {
        "iteration": 14,
        "channels": str(best_channels),
        "original_acc": "{:.2f}".format(original_acc),
        "pruned_acc": "{:.2f}".format(pruned_acc),
        "param_compression_pct": "{:.2f}".format(param_comp),
        "flops_reduction_pct": "{:.2f}".format(flops_red),
        "original_params": original_params,
        "pruned_params": pruned_params,
        "original_flops": original_flops,
        "pruned_flops": pruned_flops,
        "ga_runtime_s": "{:.2f}".format(ga_runtime),
        "kd_runtime_s": "{:.2f}".format(ft_runtime),
        "kd_T": args.kd_T,
        "kd_alpha": args.kd_alpha,
        "finetune_epochs": args.finetune_epochs,
        "acc_weight": args.acc_weight,
        "comp_weight": args.comp_weight,
        "mutation_prob": args.mutation_prob,
        "n_generations": args.n_generations,
        "seed": args.seed,
    }
    save_results(results_info, RESULT_DIR)

    print("\n" + "=" * 60)
    print("GA-14 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()