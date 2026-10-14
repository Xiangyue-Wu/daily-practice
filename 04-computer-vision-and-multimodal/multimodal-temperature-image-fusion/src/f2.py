import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import random

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Define FiLM Layer
class FiLMLayer(nn.Module):
    def __init__(self, num_features):
        super(FiLMLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

# Define FiLM Generator
class FiLMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_features):
        super(FiLMGenerator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_gamma = nn.Linear(hidden_dim, 2 * num_features)
        self.fc_beta = nn.Linear(hidden_dim, 2 * num_features)

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)
        gamma = self.fc_gamma(h_n)
        beta = self.fc_beta(h_n)
        return gamma, beta

# Define Visual Network with FiLM
class FiLMVisualNet(nn.Module):
    def __init__(self, in_channels, num_features, num_classes):
        super(FiLMVisualNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.film1 = FiLMLayer(num_features)
        
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
        self.film2 = FiLMLayer(num_features)
        
        self.conv3 = nn.Conv2d(num_features, num_features * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features * 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features * 2, num_classes)
    
    def forward(self, x, gamma1, beta1, gamma2, beta2):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.film1(x, gamma1, beta1)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.film2(x, gamma2, beta2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Custom Dataset
class FireDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path, label, temps = self.image_list[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, torch.tensor(temps, dtype=torch.float32)

# Main function
if __name__ == "__main__":
    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_dir, 'data', 'dataset')
    csv_path = os.path.join(project_dir, 'data', 'cleaned_dataset.csv')
    fire_dir = os.path.join(base_dir, 'fire', 'images')
    normal_dir = os.path.join(base_dir, 'normal')
    fire_test_dir = os.path.join(base_dir, 'fire_test', 'images')

    # Read CSV
    df = pd.read_csv(csv_path)
    train_df = df[df['type'] == 'train']
    test_df = df[df['type'] == 'test']

    # Prepare train list
    train_list = []
    for _, row in train_df.iterrows():
        co_im = row['co_im']
        temps = [row['temp1'], row['temp2'], row['temp3']]
        if 'normal_' in co_im:
            ext = '.png'
            dir_path = normal_dir
            label = 0  # no fire
        else:
            ext = '.jpg'
            dir_path = fire_dir
            label = 1  # fire
        path = os.path.join(dir_path, co_im + ext)
        if os.path.exists(path):
            train_list.append((path, label, temps))

    from collections import Counter
    train_labels = [label for _, label, _ in train_list]
    print(f"Train data label distribution: {Counter(train_labels)}")
    print(f"Total train samples: {len(train_list)}")

    # Prepare test list
    test_list = []
    for _, row in test_df.iterrows():
        co_im = row['co_im']
        temps = [row['temp1'], row['temp2'], row['temp3']]
        if 'normal_' in co_im:
            ext = '.png'
            dir_path = normal_dir
            label = 0  # no fire
        else:
            ext = '.jpg'
            dir_path = fire_test_dir
            label = 1  # fire
        path = os.path.join(dir_path, co_im + ext)
        if os.path.exists(path):
            test_list.append((path, label, temps))

    test_labels = [label for _, label, _ in test_list]
    print(f"Test data label distribution: {Counter(test_labels)}")
    print(f"Total test samples: {len(test_list)}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    train_dataset = FireDataset(train_list, transform)
    test_dataset = FireDataset(test_list, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Model parameters
    in_channels = 3
    num_features = 16
    num_classes = 2
    input_dim = 1
    hidden_dim = 32

    # Models
    film_generator = FiLMGenerator(input_dim, hidden_dim, num_features)
    visual_net = FiLMVisualNet(in_channels, num_features, num_classes)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    visual_net.to(device)
    film_generator.to(device)
    
    # Optimizer and Loss
    optimizer = optim.Adam(list(visual_net.parameters()) + list(film_generator.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training
    losses = []
    accuracies = []
    epochs = 20
    
    for epoch in range(epochs):
        visual_net.train()
        total_loss = 0
        num_batches = 0
        correct_train = 0
        total_train = 0
        
        for images, labels, temps in train_loader:
            batch_size = images.size(0)
            temps = temps.unsqueeze(2).to(device)  # (batch, 3, 1)
            images = images.to(device)
            labels = labels.to(device)

            gamma, beta = film_generator(temps)
            gamma1 = gamma[:, :num_features]
            gamma2 = gamma[:, num_features:]
            beta1 = beta[:, :num_features]
            beta2 = beta[:, num_features:]

            outputs = visual_net(images, gamma1, beta1, gamma2, beta2)
            loss = criterion(outputs, labels)
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        train_acc = 100 * correct_train / total_train if total_train > 0 else 0
        losses.append(avg_loss)
        accuracies.append(train_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # Visualization of training loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(range(1, epochs+1), losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, epochs+1), accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Evaluation
    visual_net.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, temps in test_loader:
            batch_size = images.size(0)
            temps = temps.unsqueeze(2).to(device)  # (batch, 3, 1)
            images = images.to(device)
            labels = labels.to(device)

            gamma, beta = film_generator(temps)
            gamma1 = gamma[:, :num_features]
            gamma2 = gamma[:, num_features:]
            beta1 = beta[:, :num_features]
            beta2 = beta[:, num_features:]

            outputs = visual_net(images, gamma1, beta1, gamma2, beta2)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fire

    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Test samples: {total}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fire'], 
                yticklabels=['Normal', 'Fire'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Visualization of some test predictions
    num_vis = 12
    if len(test_list) < num_vis:
        num_vis = len(test_list)
    
    # Select random samples
    vis_indices = random.sample(range(len(test_list)), num_vis)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    
    for ax in axes:
        ax.axis('off')
    
    with torch.no_grad():
        for i, idx in enumerate(vis_indices):
            if i >= len(axes):
                break
                
            path, label, temps = test_list[idx]
            image = Image.open(path).convert('RGB')
            input_img = test_transform(image).unsqueeze(0).to(device)
            temps_tensor = torch.tensor(temps, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)

            gamma, beta = film_generator(temps_tensor)
            gamma1 = gamma[:, :num_features]
            gamma2 = gamma[:, num_features:]
            beta1 = beta[:, :num_features]
            beta2 = beta[:, num_features:]

            output = visual_net(input_img, gamma1, beta1, gamma2, beta2)
            prob = F.softmax(output, dim=1)[0, 1].item()  # Probability of fire (class 1)
            
            # Create title
            true_label = "Fire" if label == 1 else "Normal"
            pred_label = "Fire" if prob >= 0.5 else "Normal"
            color = "green" if (pred_label == true_label) else "red"
            
            axes[i].imshow(image)
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\np={prob:.3f}', 
                            color=color, fontsize=10)
            axes[i].axis('on')
            axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()
    
    # Additional statistics
    print("\n=== Classification Report ===")
    print(f"Total test samples: {len(test_list)}")
    print(f"Normal samples: {sum(1 for _, label, _ in test_list if label == 0)}")
    print(f"Fire samples: {sum(1 for _, label, _ in test_list if label == 1)}")
    print(f"\nTest Accuracy: {accuracy:.2f}%")
