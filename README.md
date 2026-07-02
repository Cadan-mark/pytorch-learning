# pytorch-learning

PyTorch 学习 demo,从 autograd 原理一步步到 CNN 实战。

## 环境

```bash
pip install -r requirements.txt
```

> torch / torchvision 建议按你的系统和 CUDA 版本,到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应安装命令;只用 CPU 直接 pip install 即可。

## 建议学习顺序

按 `01 → 02 → 03 → 04` 循序渐进,每一个都在前一个基础上增加新东西。

### 01_autograd_demo.py — autograd 自动求导

理解 PyTorch 的核心机制:你只写前向,反向的梯度它自动算。

- 计算图:前向运算时自动记录每一步
- `backward()`:自动求导
- 梯度累加:为什么训练要先 `zero_grad()`
- `no_grad()`:推理时关闭梯度追踪

```bash
python 01_autograd_demo.py   # 几秒出结果,不用下载数据
```

### 02_mnist_train.py — 全连接网络,吃透训练循环

用最简单的全连接网络做 MNIST 手写数字识别,重点是训练流程。

- `nn.Module` 定义模型、`forward` 前向传播
- `Dataset` / `DataLoader` 数据加载
- 训练循环四步:前向 → 损失 → 反向 → 更新
- `zero_grad()` / `backward()` / `step()` 三连
- `model.train()` / `model.eval()` / `torch.no_grad()`

```bash
python 02_mnist_train.py   # 首次运行自动下载 MNIST(约 10MB)
```

### 03_mnist_cnn.py — CNN 卷积网络

把全连接换成 CNN,更适合图像任务,准确率可到 ~99%。

- 卷积层 `Conv2d`、池化 `MaxPool2d`
- 为什么图像任务用 CNN
- `Dropout` 防过拟合

```bash
python 03_mnist_cnn.py
```

### 04_cifar10_cnn.py — 更难数据集 + 工程技巧

CIFAR-10 彩色图片(3 通道),比 MNIST 难,并加入常用工程手段。

- 数据增强(随机翻转/裁剪)提升泛化
- `BatchNorm` 稳定训练
- 学习率调度器 `lr_scheduler`
- 保存最佳模型

```bash
python 04_cifar10_cnn.py   # 首次运行下载 CIFAR-10(约 170MB),建议有 GPU
```

## 文件一览

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_autograd_demo.py` | autograd 自动求导演示 | 入门 |
| `02_mnist_train.py` | MNIST 全连接网络 | 入门 |
| `03_mnist_cnn.py` | MNIST CNN,~99% | 进阶 |
| `04_cifar10_cnn.py` | CIFAR-10 CNN(数据增强 / BatchNorm / 调度器) | 进阶 |
