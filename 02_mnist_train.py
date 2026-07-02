"""
MNIST 手写数字识别
运行: pip install -r requirements.txt && python 02_mnist_train.py
首次运行会自动下载 MNIST 数据集。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# 模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNet().to(device)
print(model)


# 损失与优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练
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


# 评估
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

    torch.save(model.state_dict(), "mnist_model.pth")
    print("saved: mnist_model.pth")
