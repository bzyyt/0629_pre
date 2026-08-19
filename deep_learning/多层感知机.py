# %% [markdown]
# 多层感知机

# %%
# 依赖
import torch
from d2l import torch as d2l
from matplotlib_inline import backend_inline
from torch import nn

# %%
# 修复d2l版本的显示问题
d2l.use_svg_display = lambda: backend_inline.set_matplotlib_formats("svg")
# Windows + Jupyter 下使用主进程读取数据
d2l.get_dataloader_workers = lambda: 0

# %%
# 定义模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# %%
# 训练
batch_size = 256
lr = 0.1
num_epochs = 10
loss = nn.CrossEntropyLoss(reduction="mean")
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
