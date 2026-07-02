"""
MNIST + CNN 卷积神经网络
相比 02 的全连接版,用 CNN 更适合图像,准确率可到 99%。
运行: python 03_mnist_cnn.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 设备(优先 XPU / Intel GPU,其次 CUDA,都没有用 CPU)
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"device: {device}")


# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)


# CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积块 1: 1 -> 32 通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积块 2: 32 -> 64 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)     # 下采样,尺寸减半
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)    # 防过拟合
        # 两次 pool 后 28x28 -> 7x7,通道 64
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # -> [B,32,14,14]
        x = self.pool(self.relu(self.conv2(x)))   # -> [B,64,7,7]
        x = torch.flatten(x, 1)                   # -> [B,64*7*7]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = CNN().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(images):5d}/{len(train_dataset)}] "
                  f"Loss: {loss.item():.4f}")


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


if __name__ == "__main__":
    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        evaluate()

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("saved: mnist_cnn.pth")
