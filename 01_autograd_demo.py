"""
autograd 自动求导演示
运行: python 01_autograd_demo.py
"""

import torch

# --- 自动求导 ---
# requires_grad=True 追踪对 x 的运算
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x        # y = x^2 + 3x
y.backward()              # 求 dy/dx

print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"dy/dx = {x.grad.item()} (2x+3 = 7)\n")


# --- 梯度累加 ---
w = torch.tensor(1.0, requires_grad=True)
for step in range(3):
    loss = w * 2
    loss.backward()       # 不清零,梯度会累加
    print(f"step {step + 1}: w.grad = {w.grad.item()}")
print()


# --- no_grad ---
a = torch.tensor(3.0, requires_grad=True)
b = a * 2
print(f"b.requires_grad = {b.requires_grad}")

with torch.no_grad():
    c = a * 2
    print(f"c.requires_grad = {c.requires_grad}")
