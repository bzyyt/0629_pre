# %% [markdown]
# Softmax 简洁实现

# %%
# 依赖
import torch
from d2l import torch as d2l
from matplotlib_inline import backend_inline
from torch import nn

# %%
batch_size = 256
# Windows + Jupyter 下使用主进程读取数据
d2l.get_dataloader_workers = lambda: 0
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.use_svg_display = lambda: backend_inline.set_matplotlib_formats("svg")

# %%
# 初始化模型参数
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# %%
# softmax
loss = nn.CrossEntropyLoss(reduction="mean")

# %%
# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# %%
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
