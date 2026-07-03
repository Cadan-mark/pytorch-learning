# PyTorch 学习日志

记录每天学到的核心知识点，方便回头复习。每天在下面追加一节。


# Day 1 — 自动求导 / 训练循环 / nn.Module

日期：2026-07-03

今日主线：从只会跑代码、不懂原理，到能讲清模型训练的完整流程。


## 1. 自动求导（autograd）

- backward() 会自动求梯度（导数）。
- 求出的梯度存进变量的 .grad 里。
- 例：y = x * 3，y.backward() 后 x.grad 得到 3（y 对 x 的导数是 3）。


## 2. 梯度累加 和 zero_grad

- PyTorch 的梯度默认会累加，不会自动清零。
- 所以每轮训练前要先 zero_grad() 清零，防止上一轮的梯度带进来。
- 例：loss = w * 5 连做 3 次 backward() 不清零，.grad = 5, 10, 15（累加）。


## 3. no_grad（关闭梯度追踪）

- with torch.no_grad(): 会关闭梯度追踪。
- 块内算出的结果 requires_grad = False。
- 用途：推理 / 评估时用它，节约内存、加速，并防止误改参数。
- 判断技巧：看某行是否在 no_grad 块里，在里面就是不追踪(False)，在外面就是正常追踪(True)。


## 4. 训练循环（模型是怎么学习的）

一个循环，每轮做 5 件事：

- zero_grad()：清零梯度（防止累加）
- forward：前向预测，得到 output
- loss = ...：计算损失（预测和答案差多少）
- backward()：反向求梯度（存进 .grad）
- step()：按梯度更新参数（让 loss 变小）

不断循环，loss 越来越小，模型越来越准。

- loss.backward()：负责算梯度。
- optimizer.step()：负责用梯度真正更新参数。
- 顺序不能乱：先 zero_grad 清零，再 backward 求梯度，最后 step 更新（否则刚算的梯度会被清掉，参数不动）。


## 5. nn.Module（模型怎么定义）

- class XXX(nn.Module)：继承 PyTorch 的父类 nn.Module，白得一堆工具（管参数、to(device)、train/eval、自动求导等）。
- __init__：准备/定义层（备零件）。每行 self.xxx = ... 登记一个层。第一行 super().__init__() 必写。
- forward：规定数据按顺序经过哪些层（定流程）。数据 x 像接力棒，一层层被加工。
- 补充：训练循环里的 model(images) 实际上就是在跑模型的 forward。
- 类比：__init__ 是准备材料（茶、奶、珍珠），forward 是制作步骤（先倒茶，再加奶，最后放珍珠）。


## 今日能回答的面试题

- PyTorch 怎么自动求导？backward() 求梯度，存进 .grad。
- 为什么每轮训练要 zero_grad？梯度默认会累加，不清零会带上一轮的。
- no_grad 是干嘛的？什么时候用？关闭梯度追踪；推理/评估时用，省内存、加速、防误改参数。
- 讲讲训练流程？清零，前向预测，算 loss，反向求梯度，更新参数，循环，让 loss 变小。
- loss.backward() 和 optimizer.step() 区别？前者算梯度，后者用梯度更新参数。
- PyTorch 里模型怎么定义？继承 nn.Module，__init__ 定义层，forward 定义数据经过层的顺序。


## 下次可以学

- 层在算什么：nn.Linear 做什么运算、ReLU 为什么要有。
- model.train() vs model.eval()：训练和测试时模型有什么不同。
