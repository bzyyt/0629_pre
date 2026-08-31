# 0629 Pre 图像处理学习项目

2026年暑假的计算机视觉 / 深度学习主线 + 大模型入门 12 周学习成果

## 项目结构

```text
0629_pre/
├── asserts/                         # 练习使用的示例图片
│   └── week2_work/                  # 第 2 周作业图片
├── books/                           # 学习资料
├── image_processing/                # 图像处理代码与笔记
│   ├── week01_image_io.py           # 第 1 周图像读写练习
│   ├── week01_note.py               # 第 1 周学习笔记
│   ├── week02_cv_demo.py            # 第 2 周 OpenCV 练习
│   └── week02_note.py               # 第 2 周学习笔记
├── results/                         # 图像处理结果
│   ├── week1_image_io/              # 第 1 周输出
│   └── week2_cv/                    # 第 2 周输出
├── .gitattributes
├── .gitignore
├── .python-version                  # Python 版本配置
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

对于有GPU的电脑，在根目录执行：

```powershell
uv sync --extra cuda
```

没有GPU的电脑，在根目录执行：

```powershell
uv sync --extra cpu
```

依赖安装完成后，在 VS Code 中选择项目虚拟环境：

```text
.venv\Scripts\python.exe
```

## 运行方式

对于python笔记代码，在 VS Code 中打开，然后使用每个 `# %%` 单元格上方的运行按钮执行单元格。

对于作业代码，建议使用uv run命令运行

示例图片使用相对路径读取：

```python
pic1 = cv2.imread("../asserts/pic1.jpg")
```

如果提示图片读取失败，请确认当前工作目录为 `image_processing`，或根据实际工作目录调整图片路径。

在大模型有关代码运行之前，先在根目录运行：

```powershell
cp .env.example .env
```

然后在创建的.env文件里面填入所需的API密钥
