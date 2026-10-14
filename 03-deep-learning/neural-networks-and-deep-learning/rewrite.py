import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_mnist_png(root):
    # 加载MNIST_PNG数据集（目录结构：root/train/数字/, root/test/数字/）
    # 使用Grayscale(3)将灰度图复制为3通道，以复用CIFAR预训练的AlexNet特征
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'test')
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    return train_data, test_data

def create_loaders(train_data, test_data, bs=128):
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=2)
    return train_loader, test_loader

def build_model(num_classes):
    # 构建AlexNet并“重写分类层”到MNIST类数
    # 说明：模型确实“一整个加载”的，但PyTorch的模型是Python对象，我们可以直接替换其子模块
    # 这里将classifier[6]替换为新的Linear，从而“去掉原先的并增加重写的”分类层
    model = models.alexnet(pretrained=False)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

def load_ckpt(model, ckpt_path, device):
    # 加载cifarckpt.ckpt权重。若我们重写了分类层，设置strict=False即可跳过不匹配的旧分类层权重
    # 若ckpt为包含'model_state'的字典，则优先读取该键；否则作为纯state_dict加载
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model

def set_trainable_layers(model, train_classifier_only=True, train_last_conv=False):
    # 常见微调策略：冻结大部分层，只训练重写过的分类层；可选地再解冻最后一层卷积以适配新数据
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if train_last_conv:
        for p in model.features[10].parameters():
            p.requires_grad = True
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc

def visualize_curves(train_losses, test_losses, train_accs, test_accs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b', label='训练损失')
    plt.plot(epochs, test_losses, 'r', label='测试损失')
    plt.title('重写分类层-损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, 'g', label='训练准确率')
    plt.plot(epochs, test_accs, 'orange', label='测试准确率')
    plt.title('重写分类层-准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率(%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'))
    plt.close()
    return None

def build_confusion(model, loader, device, num_classes):
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf[int(t.item()), int(p.item())] += 1
    return conf

def visualize_confusion(conf, class_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total = conf.sum()
    plt.figure(figsize=(10, 9))
    im = plt.imshow(conf, cmap='Blues')
    plt.colorbar(im)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(np.arange(len(class_names)), class_names)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            val = conf[i, j]
            pct = (val / total * 100.0) if total > 0 else 0.0
            plt.text(j, i, f'{val}\n{pct:.1f}%', ha='center', va='center', color='black')
    plt.title('重写分类层-混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()
    return None

def save_model(model, out_dir, filename='mnist_rewrite_final.ckpt'):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    torch.save(model.state_dict(), save_path)
    return save_path

def train_model(model, train_loader, test_loader, num_epochs, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        v_loss, v_acc = eval_one_epoch(model, test_loader, criterion, device)
        train_losses.append(t_loss)
        test_losses.append(v_loss)
        train_accs.append(t_acc)
        test_accs.append(v_acc)
        print(f'Epoch {epoch}: 训练损失 {t_loss:.4f} | 训练准确率 {t_acc:.2f}% | 测试损失 {v_loss:.4f} | 测试准确率 {v_acc:.2f}%')
        if v_acc > best_acc:
            best_acc = v_acc
    visualize_curves(train_losses, test_losses, train_accs, test_accs, out_dir)
    class_names = list(getattr(test_loader.dataset, 'classes', [str(i) for i in range(10)]))
    conf = build_confusion(model, test_loader, device, num_classes=len(class_names))
    visualize_confusion(conf, class_names, out_dir)
    final_path = save_model(model, out_dir, filename='final_mnist_head.ckpt')
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc,
        'final_model_path': final_path
    }

def main():
    start = time.perf_counter()
    # 1. 路径与参数设置
    mnist_root = r'data/MNIST_PNG'
    ckpt_path = r'models/cifarckpt.ckpt'
    out_dir = r'data/rere'
    num_epochs = 10

    # 2. 加载数据
    train_data, test_data = load_mnist_png(mnist_root)

    # 3. 创建数据加载器
    train_loader, test_loader = create_loaders(train_data, test_data, bs=128)

    # 4. 构建模型并“重写分类层”到MNIST类别数
    num_classes = len(train_data.classes)
    model = build_model(num_classes)

    # 5. 加载CIFAR预训练权重（除了新分类层外用strict=False载入）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_ckpt(model, ckpt_path, device)

    # 6. 冻结除分类层（可选解冻最后一层卷积）
    model = set_trainable_layers(model, train_classifier_only=True, train_last_conv=False)

    # 7. 训练与验证，并保存曲线与最终模型到rere目录
    result = train_model(model, train_loader, test_loader, num_epochs=num_epochs, out_dir=out_dir)
    # 8. 输出耗时
    elapsed = time.perf_counter() - start
    print(f'总耗时：{elapsed:.2f}秒')
    return result

if __name__ == "__main__":
    # 数据导入、初始变量定义等由main内部完成
    main()

