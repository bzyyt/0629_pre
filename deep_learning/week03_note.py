# %% [markdown]
# 第三周 深度学习训练流程


# %%
# 依赖
import os

import pandas as pd
import torch

# %% [markdown]
# 张量
# 张量代表一个由数值组成的数组
# 一维张量又称向量，二维张量又称矩阵。

# %%
# 创建张量
x = torch.arange(12)
print(x)

# %%
# shape 属性可以查看张量的形状
print(x.shape)

# %%
# numel() 方法可以查看张量中元素的个数
print(x.numel())

# %%
# 使用reshape方法可以改变张量的形状
y = x.reshape(3, 4)
print(y)

# %%
# 在reshape方法中，-1表示自动计算该维度的大小
y = x.reshape(-1, 6)
print(y)

# %%
# zero()，ones()创建全零张量和全一张量
x = torch.zeros(3, 4)
print(x)

x = torch.ones(3, 4)
print(x)

# %%
# rand()创建随机张量,randn()创建标准正态分布随机张量
x = torch.rand(3, 4)
print(x)

x = torch.randn(3, 4)
print(x)

# %%
# 运算符
# 按元素运算
# 标准算数运算符都可以按元素运算，+、-、*、/、**、%等
# 一些其他的运算符也可以按元素运算，如逻辑运算符、比较运算符等
x = torch.rand(3, 4)
y = torch.rand(3, 4)
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(torch.exp(x))

# %%
# 张量的连结
# torch.cat()函数可以将多个张量沿着指定的维度连结起来
x = torch.rand(3, 4)
y = torch.rand(3, 4)
z = torch.cat((x, y), dim=0)
print(z)

# %%
# 广播机制
# 两个张量形状不一致的时候，PyTorch会自动进行广播，使得它们能够进行运算。
x = torch.rand(3, 4)
y = torch.rand(1, 4)
z = x + y
print(z)

# %%
# 索引和切片
# 张量的索引和切片机制与python的列表类似。
x = torch.rand(3, 4)
print(x[0])  # 第一行
print(x[:, 0])  # 第一列
print(x[0:2, 0:2])  # 子矩阵

# %%
# 原地操作
x = torch.rand(3, 4)
z = torch.rand(3, 4)
z[:] = x + z
print(z)


# %% [markdown]
# 数据预处理

# %%
# 生成数据集
os.makedirs(os.path.join("..", "data"), exist_ok=True)
data_file = os.path.join("..", "data", "house_tiny.csv")
with open(data_file, "w") as f:
    f.write("NumRooms,Alley,Price\n")  # 列名
    f.write("NA,Pave,127500\n")  # 每行表示一个数据样本
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

# %%
# 读取
data_file = os.path.join("..", "data", "house_tiny.csv")
data = pd.read_csv(data_file)
print(data)

# %%
# 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
# 新版pandas中，需要指定numeric_only=True来确保只对数值列进行操作
print(inputs)

# %%
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# %%
# 转换为张量
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(Y)


# %%
# 自动微分
# 标量的自动微分
x = torch.arange(4.0)
assert x.grad is not None
print(x)

# %%
# 对x启用自动微分
x.requires_grad_(True)
print(x.grad)

# %%
# 计算y = 2 * x^2
y = 2 * torch.dot(x, x)
print(y)

# %%
# 计算y对x的导数
y.backward()
print(x.grad)

# %%
# 计算另外一个函数
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# %%
# 非标量的自动微分
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)

# %%
# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)
