# %% [markdown]
# 线性回归的简洁实现


# %%
# 依赖
from typing import cast

import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

# %%
# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# %%
# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# %%
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# %%
# 定义模型
net = nn.Sequential(nn.Linear(2, 1))
# 全连接层

# %%
# 初始化模型参数
linear_layer = cast(
    nn.Linear, net[0]
)  # torch 类型存根将 Sequential[0] 推断为 Sequential|Module，此处显式断言为 nn.Linear
nn.init.normal_(linear_layer.weight, 0, 0.01)  # 权重：正态分布 N(0, 0.01)
nn.init.zeros_(linear_layer.bias)  # 偏置：全 0

# %%
# 定义损失函数
loss = nn.MSELoss()  # 均方误差损失函数

# %%
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 随机梯度下降优化器

# %%
# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {l:f}")

# %%
# 比较真实参数和训练得到的参数
w = cast(nn.Linear, net[0]).weight.data
print("w的估计误差：", true_w - w.reshape(true_w.shape))
b = cast(nn.Linear, net[0]).bias.data
print("b的估计误差：", true_b - b)

# %%
