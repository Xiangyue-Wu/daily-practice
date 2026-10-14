import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_model():
    model = models.alexnet()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    model.load_state_dict(torch.load(r'models/cifarckpt.ckpt'))
    return model

def freeze_except_last_conv(model):
    # 模型是一个整体加载的，但我们可以通过设置requires_grad来选择性地微调特定层
    # 这允许我们“冻结”大部分模型，只更新最后一层卷积层的参数
    for param in model.parameters():
        param.requires_grad = False
    # 最后一层卷积层是features[10]
    for param in model.features[10].parameters():
        param.requires_grad = True
    return model

def load_data():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为3通道
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    data_dir = r'data/MNIST_PNG'
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
    return train_data, test_data

def create_data_loaders(train_data, test_data, bs=128):
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=2)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # 只优化可训练参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    return train_losses, test_losses, train_accs, test_accs

def visualize_curves(train_losses, test_losses, train_accs, test_accs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='训练损失')
    plt.plot(epochs, test_losses, 'r', label='测试损失')
    plt.title('微调-损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()
    
    plt.figure()
    plt.plot(epochs, train_accs, 'g', label='训练准确率')
    plt.plot(epochs, test_accs, 'orange', label='测试准确率')
    plt.title('微调-准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'))
    plt.close()
    return None

def visualize_confusion_matrix(model, test_loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    num_classes = 10
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf[int(t.item()), int(p.item())] += 1
    plt.figure(figsize=(10, 9))
    im = plt.imshow(conf, cmap='Blues')
    plt.colorbar(im)
    classes = getattr(test_loader.dataset, 'classes', [str(i) for i in range(num_classes)])
    plt.xticks(np.arange(num_classes), classes, rotation=45, ha='right')
    plt.yticks(np.arange(num_classes), classes)
    total_cm = conf.sum()
    for i in range(num_classes):
        for j in range(num_classes):
            val = conf[i, j]
            pct = (val / total_cm * 100.0) if total_cm > 0 else 0.0
            plt.text(j, i, f'{val}\n{pct:.1f}%', ha='center', va='center', color='black')
    plt.title('微调-混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()
    return None

def save_model(model, path):
    torch.save(model.state_dict(), path)
    return None

def main():
    start = time.perf_counter()
    # 1. 加载模型
    model = load_model()
    
    # 2. 冻结除最后一层卷积层外的所有层
    model = freeze_except_last_conv(model)
    
    # 3. 加载数据
    train_data, test_data = load_data()
    
    # 4. 创建数据加载器
    train_loader, test_loader = create_data_loaders(train_data, test_data)
    
    # 5. 训练模型
    train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader)
    
    # 6. 可视化曲线
    out_dir = r'data/refi'
    visualize_curves(train_losses, test_losses, train_accs, test_accs, out_dir)
    visualize_confusion_matrix(model, test_loader, out_dir)
    
    # 7. 保存最终模型
    save_path = os.path.join(out_dir, 'finetuned_mnist.ckpt')
    save_model(model, save_path)
    # 8. 输出耗时
    elapsed = time.perf_counter() - start
    print(f'总耗时：{elapsed:.2f}秒')
    
    return None

if __name__ == "__main__":
    main()
