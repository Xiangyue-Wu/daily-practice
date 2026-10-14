# CIFAR-10 图像分类 - 优化版本 try2.py
# 基于try1.py的优化改进

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, ColorJitter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np

def load_data():
    """加载CIFAR-10数据集，添加数据增强"""
    # 训练时的数据增强
    train_transform = Compose([
        RandomCrop(32, padding=4),  # 随机裁剪，填充4像素
        RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 测试时只做标准化
    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载训练和测试数据集
    train_data = datasets.ImageFolder('./cifar10/train', transform=train_transform)
    test_data = datasets.ImageFolder('./cifar10/test', transform=test_transform)
    
    return train_data, test_data

def create_data_loaders(train_data, test_data):
    """创建数据加载器"""
    bs = 128  # 增大批次大小
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, bs

class ImprovedCIFAR10CNN(nn.Module):
    """改进的CIFAR-10卷积神经网络"""
    def __init__(self):
        super(ImprovedCIFAR10CNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 第二个卷积块
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # 第三个卷积块
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout层
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层
        # 经过3次池化后：32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        
        # 第二个卷积块
        x = self.pool(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        
        # 第三个卷积块
        x = self.pool(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))
        
        # Dropout
        x = self.dropout1(x)
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def train_model(model, train_loader, test_loader, num_epochs=25):
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
        
        # 更新学习率
        scheduler.step()
        
        # 计算训练准确率
        train_acc = 100 * correct_train / total_train
        
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
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'\nEpoch {epoch} 结果:')
        print(f'训练准确率: {train_acc:.2f}%')
        print(f'测试准确率: {test_acc:.2f}%')
        print(f'测试损失: {avg_test_loss:.4f}')
        print(f'当前学习率: {scheduler.get_last_lr()[0]:.6f}')
        print(f'最佳测试准确率: {best_acc:.2f}%\n')
    
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
    
    # 4.创建改进的模型
    model = ImprovedCIFAR10CNN()
    print('模型结构:')
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n总参数数量: {total_params:,}')
    print(f'可训练参数数量: {trainable_params:,}')
    
    # 5.训练模型
    best_acc = train_model(model, train_loader, test_loader, num_epochs=25)
    
    # 6.加载最佳模型并预测展示
    model.load_state_dict(torch.load('best_model.pth'))
    predict_and_show(model, test_loader)
    
    print(f"\n最终最佳测试准确率: {best_acc:.2f}%")
    
    return best_acc

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    final_acc = main()