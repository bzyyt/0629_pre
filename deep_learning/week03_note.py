# %% [markdown]
# 第三周 深度学习训练流程


# %%
# 依赖
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
# 标准算数运算符都可以暗元素运算，+、-、*、/、**、%等
# 一些其他的运算符也可以按元素运算，如逻辑运算符、比较运算符等
x = torch.rand(3, 4)
y = torch.rand(3, 4)
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(torch.exp(x))

# %%
