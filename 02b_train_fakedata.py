"""
训练流程 demo(假数据版,无需 torchvision)
和 02 完全相同的训练流程,只是把数据换成随机生成的假数据,
因此不需要 torchvision、不需要联网下载,任何环境都能直接跑。

注意:假数据没有规律,所以准确率会停在 ~10%(10 类瞎猜),这是正常的。
本脚本的目的是让你跑通并理解“训练一个模型”的完整流程,不是追求准确率。
运行: python 02b_train_fakedata.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# 设备(优先 XPU / Intel GPU,其次 CUDA,都没有用 CPU)
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"device: {device}")


# 假数据:模拟 MNIST 的形状
# 图片: 6000 张, 每张 1x28x28(和 MNIST 一样);标签: 0-9 随机
N = 6000
fake_images = torch.randn(N, 1, 28, 28)
fake_labels = torch.randint(0, 10, (N,))

dataset = TensorDataset(fake_images, fake_labels)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# 模型(和 02 一样的全连接网络)
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


# 训练一个 epoch
def train_one_epoch(epoch):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 训练循环四步:前向 -> 损失 -> 反向 -> 更新
        outputs = model(images)              # 1. 前向传播
        loss = loss_fn(outputs, labels)      # 2. 算损失

        optimizer.zero_grad()                # 3a. 梯度清零(否则会累加)
        loss.backward()                      # 3b. 反向传播,自动求梯度
        optimizer.step()                     # 4.  用梯度更新参数

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(images):5d}/{N}] "
                  f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)

    torch.save(model.state_dict(), "fakedata_model.pth")
    print("saved: fakedata_model.pth")
    print("\n提示:假数据没有规律,loss 不会明显下降、准确率约 10%,属于正常现象。")
    print("装好 torchvision 后,跑 02_mnist_train.py 用真实数据,就能看到 loss 下降、准确率上升。")
