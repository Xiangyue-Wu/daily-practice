# CIFAR-10 图像分类 - 使用AlexNet

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, ColorJitter, Resize
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载CIFAR-10数据集，添加数据增强并调整大小以适应AlexNet"""
    # 训练时的数据增强
    train_transform = Compose([
        Resize(224),  # 先调整大小
        RandomCrop(224, padding=28),  # 随机裁剪，调整填充以匹配比例
        RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 测试时只做标准化
    test_transform = Compose([
        Resize(224),  # 调整大小
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载训练和测试数据集
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    return train_data, test_data

def create_data_loaders(train_data, test_data):
    """创建数据加载器"""
    bs = 128  # 增大批次大小
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, bs

def get_model():
    """获取并修改AlexNet模型"""
    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    return model

def train_model(model, train_loader, test_loader, num_epochs=10):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'使用设备: {device}')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加权重衰减
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    print("================== 开始训练 ==================")
    
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # 每100个批次输出一次
            if batch_idx % 100 == 0:
                print(f'训练 Epoch: {epoch:3d} [{batch_idx * len(images):5d}/{len(train_loader.dataset):5d} '
                      f'({100 * batch_idx / len(train_loader):3.0f}%)]\t损失值: {loss.item():.6f}')
        
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        train_acc = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # 测试阶段
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_acc = 100 * correct_test / total_test
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # torch.save(model.state_dict(), 'best_model.pth')
            torch.save(model.state_dict(), os.path.join(r'models/model', 'cifarckpt.ckpt'))
        
        print(f'\nEpoch {epoch} 结果:')
        print(f'训练准确率: {train_acc:.2f}%')
        print(f'测试准确率: {test_acc:.2f}%')
        print(f'测试损失: {avg_test_loss:.4f}')
        print(f'当前学习率: {scheduler.get_last_lr()[0]:.6f}')
        print(f'最佳测试准确率: {best_acc:.2f}%\n')
    
    out_dir = r'data/re2'
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, color='blue')
    plt.plot(epochs, test_losses, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练/测试损失曲线')
    plt.legend(['训练损失', '测试损失'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curves.png'))
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, color='green', linestyle='--')
    plt.plot(epochs, test_accs, color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('准确率(%)')
    plt.title('训练/测试准确率曲线')
    plt.legend(['训练准确率', '测试准确率'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_curves.png'))
    plt.close()
    cn_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    conf = np.zeros((10, 10), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf[int(t.item()), int(p.item())] += 1
    total_cm = conf.sum()
    plt.figure(figsize=(10, 9))
    im = plt.imshow(conf, cmap='Blues')
    plt.colorbar(im)
    plt.xticks(np.arange(10), cn_names, rotation=45, ha='right')
    plt.yticks(np.arange(10), cn_names)
    for i in range(10):
        for j in range(10):
            val = conf[i, j]
            pct = (val / total_cm * 100.0) if total_cm > 0 else 0.0
            plt.text(j, i, f'{val}\n{pct:.1f}%', ha='center', va='center', color='black')
    plt.title('CIFAR-10 混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()
    
    return best_acc

def predict_and_show(model, test_loader):
    """预测并展示结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 获取一批测试数据
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 显示前8张图片的预测结果
    plt.figure(figsize=(12, 8))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        
        # 反标准化图片
        img = images[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        plt.imshow(img.permute(1, 2, 0))
        
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predicted[i].item()]
        
        # 如果预测正确显示绿色，错误显示红色
        color = 'green' if labels[i].item() == predicted[i].item() else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return None

def main():
    # 1.加载数据集
    train_data, test_data = load_data()
    
    # 2.创建数据加载器
    train_loader, test_loader, bs = create_data_loaders(train_data, test_data)
    
    # 3.输出数据集信息
    print('训练数据集数量：', len(train_loader.dataset))
    print('测试数据集数量：', len(test_loader.dataset))
    
    # 4.创建模型
    model = get_model()
    print('模型结构:')
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n总参数数量: {total_params:,}')
    print(f'可训练参数数量: {trainable_params:,}')
    
    # 5.训练模型
    best_acc = train_model(model, train_loader, test_loader, num_epochs=10)
    
    # 6.加载最佳模型并预测展示
    # model.load_state_dict(torch.load('best_model.pth'))
    model.load_state_dict(torch.load(os.path.join(r'models/model', 'cifarckpt.ckpt')))
    predict_and_show(model, test_loader)
    
    print(f"\n最终最佳测试准确率: {best_acc:.2f}%")
    
    return best_acc

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    final_acc = main()
