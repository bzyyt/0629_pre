# %% [markdown]
# 线性回归
# 线性回归用于预测一个函数，输入与输出存在一个线性关系
# $$y = wx + b$$
# 线性回归可以看成单层的神经网络

# %% [markdown]
# 损失函数
# 损失函数用于衡量预测与真实之间的差距
# 一般使用平方差损失函数
# $$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
# 一般目标是最小化损失函数

# %% [markdown]
# 随机梯度下降
# 梯度下降最简单的表示就是计算损失函数相对于模型参数的导数
# 由于这种方法一般比较慢，通常使用的时候会抽取小批量的样本，即小批量随机梯度下降
# 学习率不能太大，也不能太小

# %%
# 线性回归从零开始
# 依赖
import random

import matplotlib_inline.backend_inline
import torch
from d2l import torch as d2l

# 兼容补丁：d2l 0.17.0 调用的 display.set_matplotlib_formats 在 IPython 8+ 已被移除
from IPython import display

if not hasattr(display, "set_matplotlib_formats"):
    display.set_matplotlib_formats = (  # type: ignore[attr-defined]
        matplotlib_inline.backend_inline.set_matplotlib_formats
    )


# %%
# 生成数据集
def synthetic_data(w, b, num_examples):  # @save
    """生成y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# %%
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# %%
print("features:", features[0], "\nlabel:", labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)


# %%
# 读取数据集
def data_iter(batch_size, features, labels):
    # 每个批次的样本数量，特征矩阵，标签向量
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 原地打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)]
        )  # @save
        yield features[batch_indices], labels[batch_indices]


# %%
# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# %%
# 定义模型
def linreg(X, w, b):  # @save
    """线性回归模型"""
    return torch.matmul(X, w) + b
    # 返回矢量乘法


# %%
# 定义损失函数
def squared_loss(y_hat, y):  # @save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    # 返回矢量


# %%
# 定义优化算法
def sgd(params, lr, batch_size):  # @save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# %%
# 训练
# 初始化参数->计算梯度->更新参数->重复
lr = 0.03  # 学习率
num_epochs = 3  # 迭代次数
net = linreg  # 模型
loss = squared_loss  # 损失函数

# %%
for epoch in range(num_epochs):
    for X, y in data_iter(10, features, labels):
        l = loss(net(X, w, b), y)  # 计算损失
        l.sum().backward()  # 反向传播计算梯度
        sgd([w, b], lr, batch_size=10)  # 使用小批量随机梯度下降迭代模型参数

    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

# %%
# 比较真实参数和训练得到的参数
print(f"w的估计误差: {true_w - w.reshape(true_w.shape)}")
print(f"b的估计误差: {true_b - b}")

# %%
