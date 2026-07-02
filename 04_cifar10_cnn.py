"""
CIFAR-10 + CNN 进阶版
比 MNIST 难:彩色图片(3 通道)、图像更复杂。
额外包含工程细节:数据增强、学习率调度、保存最佳模型。
运行: python 04_cifar10_cnn.py
首次运行会下载 CIFAR-10 数据集(约 170MB)。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# 数据增强:训练集随机翻转/裁剪,提升泛化能力
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


# CNN 模型(带 BatchNorm)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # 三次 pool: 32 -> 16 -> 8 -> 4,通道 128
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # -> [B,32,16,16]
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # -> [B,64,8,8]
        x = self.pool(self.relu(self.bn3(self.conv3(x))))   # -> [B,128,4,4]
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = CNN().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 学习率调度:每 5 个 epoch 学习率乘 0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(images):5d}/{len(train_dataset)}] "
                  f"Loss: {loss.item():.4f}")

    return running_loss / len(train_loader)


def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"accuracy: {accuracy:.2f}% ({correct}/{total})\n")
    return accuracy


if __name__ == "__main__":
    EPOCHS = 15
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        acc = evaluate()
        scheduler.step()   # 更新学习率

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "cifar10_best.pth")
            print(f"new best: {acc:.2f}%, saved cifar10_best.pth\n")

    print(f"training done, best accuracy: {best_acc:.2f}%")
