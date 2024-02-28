# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 2.pytorch实现.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/6/9 19:13
@Function：
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time

# 数据集
x0 = np.random.randn(5000, 1)   # 生成横坐标随机数组，5000个 维度为1的数组
x1 = np.random.randn(5000, 1)
x2 = 2*x0+2*np.random.rand(5000, 1)+1   # 在生成的随机数组的基础上，对上下两部分进行扩充
x3 = 2*x1-2*np.random.rand(5000, 1)-1
y0 = np.ones((5000, 1))         # 生成横坐标随机数组
y1 = -y0        # 生成标签
n = np.vstack((x0, x1))         # 垂直堆叠数组，形成坐标
m = np.vstack((x2, x3))
x_train = np.hstack((n, m))     # 水平堆叠数组，形成坐标
x_target = np.vstack((y0, y1))  # 竖直标签

X_train = torch.from_numpy(x_train)     # 转换tensor格式
# print(X_train.dtype)
X_target = torch.from_numpy(x_target)
inputs = torch.as_tensor(X_train, dtype=torch.float32)
targets = torch.as_tensor(X_target, dtype=torch.float32)

# 定义模型
model = nn.Linear(2, 1)     # 设置全连接层，输入输出为四维张量，相当于调用一个wx+b=y 函数，其中的w，b都自动给出指定格式
opt = torch.optim.SGD(model.parameters(), lr=0.001)  # 选择优化器 SGD为随机梯度下降函数
# 训练
px = []
py = []
start = time.perf_counter()  # 是返回程序开始运行到调用这个语句所经历的时间
for i in range(20000):
    L2 = torch.matmul(model.weight, model.weight.T)     # w*wT
    # print(L2)
    classification_term = torch.mean(torch.maximum(torch.tensor((0.)), 1.-model(inputs)*targets))   # 按行求平均值，返回所有元素平均值
    # model(inputs)*targets与0相比较，（0转化为tensor格式）

    # print(classification_term)
    Loss = L2+classification_term
    # print(Loss)
    Loss.backward()     # 梯度下降函数 反向传播计算得到每个参数的梯度值
    opt.step()          # 用opt.step()实现model中的w和b的改变 梯度下降执行一步参数更新
    opt.zero_grad()     # 将梯度归零
    # px.append(i)
    # py.append(Loss.item())
elapsed = (time.perf_counter()-start)
print(f'time use :{elapsed}')

[[a1, a2]] = model.weight
# print(model.bias)
[b] = model.bias
# 可视化
plt.plot(x_train[:5000, 0], x_train[:5000, 1], "ob")
plt.plot(x_train[5000:, 0], x_train[5000:, 1], "or")
p = -(a2.item()/a1.item())*x_train[:, 0]-b.item()/a1.item()
p2 = p-1
p3 = p+1
plt.plot(x_train[:, 0], p)
plt.plot(x_train[:, 0], p2)
plt.plot(x_train[:, 0], p3)
plt.show()
