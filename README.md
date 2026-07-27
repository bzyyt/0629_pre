# 0629 Pre 图像处理学习项目

2026年暑假的计算机视觉 / 深度学习主线 + 大模型入门 12 周学习成果

## 项目结构

```text
0629_pre/
├── asserts/                         # 示例图片
├── books/                           # 学习资料
├── image_processing/
│   ├── week01_image_io.py           # 图像处理练习代码
│   └── week01_work.py               # 练习与作业
├── LICENSE
├── pyproject.toml                   # 项目和依赖配置
├── uv.lock                          # uv 依赖锁定文件
└── README.md
```

## 环境要求

- Python 3.11 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 推荐使用 VS Code
- 推荐安装 VS Code 的 Python 和 Jupyter 扩展

## 安装依赖

在项目根目录执行：

```powershell
uv sync
```

依赖安装完成后，在 VS Code 中选择项目虚拟环境：

```text
.venv\Scripts\python.exe
```

## 运行方式

对于python代码，在 VS Code 中打开，然后使用每个 `# %%` 单元格上方的运行按钮执行单元格。

示例图片使用相对路径读取：

```python
pic1 = cv2.imread("../asserts/pic1.jpg")
```

如果提示图片读取失败，请确认当前工作目录为 `image_processing`，或根据实际工作目录调整图片路径。

