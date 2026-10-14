# CIFAR-10 图像分类 - 模型使用和可视化分析
# 直接使用try2.py训练好的最佳模型进行预测和可视化分析

import warnings
# 忽略torchvision图像扩展警告
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
warnings.filterwarnings("ignore", category=UserWarning)

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import font_manager
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 设置中文字体支持 - 解决中文显示问题
def setup_chinese_font():
    """
    设置中文字体支持，解决matplotlib中文显示问题
    """
    try:
        # 方法1: 使用系统字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # print("中文支持设置成功：SimHei字体")
    except:
        try:
            # 方法2: 使用matplotlib的字体管理器
            font_path = font_manager.findfont(font_manager.FontProperties(family=['SimHei', 'Microsoft YaHei']))
            if font_path:
                plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
            print("中文支持设置成功：通过字体管理器")
        except:
            # 方法3: 备用方案
            plt.rcParams['font.family'] = ['DejaVu Sans']
            print("使用备用字体，中文可能显示为方框")

# 在程序开始时调用字体设置
setup_chinese_font()

class ImprovedCIFAR10CNN(nn.Module):
    """
    改进的CIFAR-10卷积神经网络
    与try2.py中的模型结构完全一致
    """
    def __init__(self):
        super(ImprovedCIFAR10CNN, self).__init__()
        
        # 第一个卷积块：提取低级特征（边缘、颜色等）
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 第二个卷积块：提取中级特征（纹理、形状部件）
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # 第三个卷积块：提取高级特征（物体部件、整体形状）
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # 池化层：下采样，减少特征图尺寸，增加感受野
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout层：随机失活，防止过拟合
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层：将特征映射到类别概率
        # 经过3次池化后：32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用He初始化 - 特别适合ReLU激活函数"""
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
        """前向传播过程：定义数据从输入到输出的流动路径"""
        # 第一个卷积块：卷积->BN->ReLU->卷积->BN->ReLU->池化
        x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        
        # 第二个卷积块
        x = self.pool(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        
        # 第三个卷积块
        x = self.pool(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))
        
        # Dropout：只在训练时起作用
        x = self.dropout1(x)
        
        # 展平：将多维特征图转换为一维向量，供全连接层使用
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def load_test_data():
    """
    加载测试数据集
    """
    # 测试时只做标准化 - 与训练时保持一致
    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载测试数据集 - 修正路径
    test_data = datasets.ImageFolder('../cifar10/test', transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    return test_loader

def load_trained_model(model_path='best_model.pth'):
    """
    加载训练好的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型实例
    model = ImprovedCIFAR10CNN()
    
    # 加载模型参数
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # 设置为评估模式
        print(f"成功加载模型: {model_path}")
        print(f"使用设备: {device}")
        return model, device
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        return None, None
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None, None

def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 保存预测结果和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 计算每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 计算总体准确率
    overall_accuracy = 100 * correct / total
    print(f'\n总体测试准确率: {overall_accuracy:.2f}%')
    
    # 输出各类别准确率
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print('\n各类别准确率:')
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]:12s}: {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
    
    return overall_accuracy, all_predictions, all_labels, class_correct, class_total

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵 - CIFAR-10分类结果', fontsize=16, pad=20)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print("混淆矩阵已保存到 visualizations/confusion_matrix.png")

def plot_class_accuracy_bar(class_correct, class_total, class_names):
    """
    绘制各类别准确率柱状图
    """
    accuracies = []
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(class_names, accuracies, color='skyblue', alpha=0.8)
    
    # 在柱状图上显示数值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.title('各类别准确率对比', fontsize=16, pad=20)
    plt.xlabel('类别', fontsize=14)
    plt.ylabel('准确率 (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('visualizations/class_accuracy_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("类别准确率对比图已保存到 visualizations/class_accuracy_bar.png")

def plot_prediction_confidence_distribution(model, test_loader, device):
    """
    分析预测置信度分布
    """
    model.eval()
    confidences = []
    correct_confidences = []
    incorrect_confidences = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            
            confidences.extend(max_probs.cpu().numpy())
            
            # 分别记录正确和错误预测的置信度
            correct_mask = (predicted == labels)
            correct_confidences.extend(max_probs[correct_mask].cpu().numpy())
            incorrect_confidences.extend(max_probs[~correct_mask].cpu().numpy())
    
    plt.figure(figsize=(15, 5))
    
    # 总体置信度分布
    plt.subplot(1, 3, 1)
    plt.hist(confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('总体预测置信度分布', fontsize=14)
    plt.xlabel('置信度')
    plt.ylabel('频次')
    plt.grid(alpha=0.3)
    
    # 正确预测的置信度分布
    plt.subplot(1, 3, 2)
    plt.hist(correct_confidences, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('正确预测置信度分布', fontsize=14)
    plt.xlabel('置信度')
    plt.ylabel('频次')
    plt.grid(alpha=0.3)
    
    # 错误预测的置信度分布
    plt.subplot(1, 3, 3)
    plt.hist(incorrect_confidences, bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.title('错误预测置信度分布', fontsize=14)
    plt.xlabel('置信度')
    plt.ylabel('频次')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("预测置信度分布图已保存到 visualizations/confidence_distribution.png")
    
    print(f"平均置信度: {np.mean(confidences):.3f}")
    print(f"正确预测平均置信度: {np.mean(correct_confidences):.3f}")
    print(f"错误预测平均置信度: {np.mean(incorrect_confidences):.3f}")

def analyze_model_predictions(model, test_loader, device):
    """
    分析模型预测流向
    """
    model.eval()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 创建预测流向矩阵
    flow_matrix = np.zeros((10, 10))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for true_label, pred_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                flow_matrix[true_label][pred_label] += 1
    
    # 绘制预测流向热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(flow_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('模型预测流向分析', fontsize=16, pad=20)
    plt.xlabel('预测类别', fontsize=14)
    plt.ylabel('真实类别', fontsize=14)
    plt.tight_layout()
    
    plt.savefig('visualizations/prediction_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("预测流向分析图已保存到 visualizations/prediction_flow.png")

def plot_top_errors(model, test_loader, device, top_k=16):
    """
    显示最容易出错的预测样本
    """
    model.eval()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    errors = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            
            # 找出错误预测
            incorrect_mask = (predicted != labels)
            if incorrect_mask.sum() > 0:
                incorrect_images = images[incorrect_mask]
                incorrect_labels = labels[incorrect_mask]
                incorrect_preds = predicted[incorrect_mask]
                incorrect_probs = max_probs[incorrect_mask]
                
                for i in range(len(incorrect_images)):
                    errors.append({
                        'image': incorrect_images[i].cpu(),
                        'true_label': incorrect_labels[i].cpu().item(),
                        'pred_label': incorrect_preds[i].cpu().item(),
                        'confidence': incorrect_probs[i].cpu().item()
                    })
    
    # 按置信度排序，显示最"自信"的错误预测
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    plt.figure(figsize=(16, 12))
    for i in range(min(top_k, len(errors))):
        plt.subplot(4, 4, i + 1)
        
        # 反标准化图片
        img = errors[i]['image']
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        plt.imshow(img.permute(1, 2, 0))
        
        true_name = class_names[errors[i]['true_label']]
        pred_name = class_names[errors[i]['pred_label']]
        confidence = errors[i]['confidence']
        
        plt.title(f'真实: {true_name}\n预测: {pred_name}\n置信度: {confidence:.3f}', 
                 fontsize=10, color='red')
        plt.axis('off')
    
    plt.suptitle('最容易出错的预测样本 (按置信度排序)', fontsize=16, y=0.95)
    plt.tight_layout()
    
    plt.savefig('visualizations/top_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("错误样本分析图已保存到 visualizations/top_errors.png")

def predict_and_visualize(model, test_loader, device, num_images=12):
    """
    预测并可视化结果 - 显示正确和错误的预测
    """
    model.eval()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        max_probs, predicted = torch.max(probabilities, 1)
    
    # 显示预测结果
    plt.figure(figsize=(16, 12))
    for i in range(min(num_images, len(images))):
        plt.subplot(3, 4, i + 1)
        
        # 反标准化图片
        img = images[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        plt.imshow(img.permute(1, 2, 0))
        
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predicted[i].item()]
        confidence = max_probs[i].item()
        
        # 如果预测正确显示绿色，错误显示红色
        color = 'green' if labels[i].item() == predicted[i].item() else 'red'
        plt.title(f'真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.3f}', 
                 color=color, fontsize=11)
        plt.axis('off')
    
    plt.suptitle('CIFAR-10测试集预测结果可视化', fontsize=16, y=0.95)
    plt.tight_layout()
    
    plt.savefig('visualizations/prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("预测结果可视化已保存到 visualizations/prediction_visualization.png")

def main():
    """主函数"""
    print("=" * 60)
    print("使用try2.py训练好的模型进行全面分析和可视化")
    print("=" * 60)
    
    try:
        # 1. 加载训练好的模型
        model, device = load_trained_model('best_model.pth')
        if model is None:
            return None
        
        # 2. 加载测试数据
        test_loader = load_test_data()
        print(f"测试数据集数量: {len(test_loader.dataset)}")
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 3. 评估模型性能
        print("\n正在评估模型性能...")
        overall_accuracy, all_predictions, all_labels, class_correct, class_total = evaluate_model(model, test_loader, device)
        
        # 4. 绘制混淆矩阵
        print("\n生成混淆矩阵...")
        plot_confusion_matrix(all_labels, all_predictions, class_names)
        
        # 5. 绘制各类别准确率柱状图
        print("\n生成类别准确率对比图...")
        plot_class_accuracy_bar(class_correct, class_total, class_names)
        
        # 6. 分析预测置信度分布
        print("\n分析预测置信度分布...")
        plot_prediction_confidence_distribution(model, test_loader, device)
        
        # 7. 分析模型预测流向
        print("\n生成预测流向分析...")
        analyze_model_predictions(model, test_loader, device)
        
        # 8. 显示最容易出错的样本
        print("\n分析最容易出错的预测样本...")
        plot_top_errors(model, test_loader, device)
        
        # 9. 可视化正确和错误的预测样本
        print("\n显示预测结果可视化...")
        predict_and_visualize(model, test_loader, device)
        
        print(f"\n" + "=" * 60)
        print(f"模型全面分析完成！最终准确率: {overall_accuracy:.2f}%")
        print("=" * 60)
        print("\n所有可视化图片已保存到 visualizations/ 文件夹中:")
        print("- confusion_matrix.png: 混淆矩阵")
        print("- class_accuracy_bar.png: 各类别准确率对比")
        print("- confidence_distribution.png: 预测置信度分布")
        print("- prediction_flow.png: 预测流向分析")
        print("- top_errors.png: 最容易出错的样本")
        print("- prediction_visualization.png: 预测结果可视化")
        
        return overall_accuracy
        
    except FileNotFoundError:
        print("错误: 找不到模型文件 'best_model.pth'")
        print("请确保try2.py已经训练完成并保存了模型文件")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    final_acc = main()