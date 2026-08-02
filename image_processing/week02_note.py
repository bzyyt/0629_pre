# %% [markdown]
# 第二周 增强、边缘与传统实验


# %%
# 依赖
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# %%
# 图像灰度化
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()


# %% [markdown]
# 阈值分割
# 将图像转换为二值图像，即每个像素点的值要么是0，要么是255

# %%
# 二进制阈值化
# 即大于某个值的像素变为255，小于某个值的像素变为0
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 反二进制阈值化
# 即大于某个值的像素变为0，小于某个值的像素变为255
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 截断阈值化
# 即超过阈值的像素变为阈值，小于阈值的像素保持不变
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 阈值化为0
# 即大于阈值的像素保持不变，小于阈值的像素变为0
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 反阈值化为0
# 即大于阈值的像素变为0，小于阈值的像素保持不变
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 自适应阈值
# 根据图像每个局部区域的统计信息来确定阈值
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 自适应高斯阈值
# 根据图像每个局部区域的高斯加权平均值来确定阈值
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()


# %% [markdown]
# 形态学操作
# 通过对图像进行结构元素的卷积操作，实现对图像的处理

# %%
# 图像腐蚀
# 腐蚀膨胀是用来对二值图像进行的操作
# 腐蚀操作是将图像中的前景区域变小，背景区域变大
# 在卷积核内所有像素点都为1时，中心像素点的值为1，其余为0
# 对于灰度图，腐蚀操作取邻域内的较小值
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.erode(image, kernel, iterations=1)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 图像膨胀
# 膨胀操作是将图像中的前景区域变大，背景区域变小
# 在卷积核内所有像素点都为0时，中心像素点的值为0，其余为1
# 对于灰度图，膨胀操作取邻域内的较大值
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.dilate(image, kernel, iterations=1)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 图像开运算
# 开运算 = 腐蚀 + 膨胀
# 先腐蚀，再膨胀
# 用于去除较小的白色噪点，保留大块的白色区域
# 较小的部分被腐蚀消失，无法膨胀回来
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 图像闭运算
# 闭运算 = 膨胀 + 腐蚀
# 先膨胀，再腐蚀
# 用于填充图像中的小孔洞，连接相邻的白色区域
# 较小的部分被膨胀填充，无法腐蚀消失
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
plt.imshow(thresh)
plt.axis("off")
plt.show()

# %%
# 图像梯度
# 形态学梯度 = 膨胀 - 腐蚀
# 主要作用是提取物体的边缘
# 膨胀之后白色区域扩大，腐蚀之后白色区域缩小，两者相减得到边缘
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
grad = cv2.cvtColor(grad, cv2.COLOR_GRAY2RGB)
plt.imshow(grad)
plt.axis("off")
plt.show()

# %%
# 顶帽运算
# 原图 - 开区间
# 提取小尺寸的亮点区域
# 找回开运算中被腐蚀掉的亮点
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
result = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 底帽运算
# 原图 - 闭区间
# 提取小尺寸的暗点区域
# 找回闭运算中被膨胀掉的暗点
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
result = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()


# %% [markdown]
# 直方图与图像增强
# 直方图是用来描述图像中像素值分布的统计图，横轴表示像素值，纵轴表示像素数量
# 直方图关心每种像素值出现的频率

# %%
# 灰度直方图
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
result = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.plot(result)
plt.xlim((0, 256))
plt.xlabel("灰度值")
plt.ylabel("像素数量")
plt.show()

# %%
# 彩色直方图
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
color = ("r", "g", "b")
for i, col in enumerate(color):
    result = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(result, color=col, label=col.upper())
plt.xlim((0, 256))
plt.xlabel("颜色值")
plt.ylabel("像素数量")
plt.legend()
plt.show()

# %%
# 掩膜直方图
# 一块区域的直方图
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
mask = np.zeros(image.shape[:2], np.uint8)
mask[100:300, 100:300] = 255
color = ("r", "g", "b")
for i, col in enumerate(color):
    result = cv2.calcHist([image], [i], mask, [256], [0, 256])
    plt.plot(result, color=col, label=col.upper())
plt.xlim((0, 256))
plt.xlabel("灰度值")
plt.ylabel("像素数量")
plt.show()

# %%
# HS直方图
# HSV空间里的二维直方图，横轴表示色调H，纵轴表示饱和度S
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2HSV)
result = cv2.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(
    result,
    interpolation="nearest",
    aspect="auto",
    origin="lower",
    extent=(0.0, 256.0, 0.0, 180.0),
)
plt.xlabel("色调")
plt.ylabel("饱和度")
plt.colorbar(label="Pixel count")
plt.show()

# %%
# 直方图均衡化
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(image)
equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
plt.imshow(equ)
plt.axis("off")
plt.show()

# %%
# 局部直方图均衡化
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
result = clahe.apply(image)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()


# %%
# 自动色彩均衡化
# 将当前像素与其邻域内的像素进行比较，计算出一个响应值，然后根据这个响应值来调整当前像素的亮度
def auto_color_balance(
    image: np.ndarray,
    radius: int = 5,
    alpha: float = 8.0,
    clip_percent: float = 0.5,
) -> np.ndarray:

    if image is None:
        raise ValueError("输入图像为空")

    src = image.astype(np.float32) / 255.0
    height, width = src.shape[:2]

    # 镜像拓展边界
    padded = cv2.copyMakeBorder(
        src, radius, radius, radius, radius, cv2.BORDER_REFLECT_101
    )
    response = np.zeros_like(src, dtype=np.float32)
    weight_sum = 0.0

    # 将当前像素与周围像素进行比较
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            distance = np.hypot(dy, dx)
            weight = 1.0 / distance
            shifted = padded[
                radius + dy : radius + dy + height, radius + dx : radius + dx + width
            ]
            difference = np.clip(alpha * (src - shifted), -1.0, 1.0)
            response += weight * difference
            weight_sum += weight
    response /= weight_sum

    strength = 0.3
    result = np.clip(src + strength * response, 0.0, 1.0)

    return np.round(result * 255.0).astype(np.uint8)


pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
result = auto_color_balance(pic1, radius=15, alpha=2.5, clip_percent=1.0)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()


# %% [markdown]
# 图像空间域滤波

# %%
# 均值滤波
# 用一个固定大小的卷积核在图像上滑动，计算卷积核覆盖区域内的像素平均值，然后将这个平均值作为新的像素值
# 均值滤波可以有效地去除图像中的随机噪声，但会导致图像边缘模糊
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
blur = cv2.blur(pic1, (5, 5))
image = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 方框滤波
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
box = cv2.boxFilter(pic1, -1, (3, 3), normalize=False)
image = cv2.cvtColor(box, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 高斯滤波
# 高斯滤波是一种线性滤波器，它使用高斯函数作为卷积核，对图像进行平滑处理
# 高斯滤波可以有效地去除图像中的高斯噪声，同时保持图像的边缘信息
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
gaussian = cv2.GaussianBlur(pic1, (5, 5), 0)
image = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 中值滤波
# 中值滤波是一种非线性滤波器，它用卷积核覆盖区域内的中值来替换中心像素的值
# 中值滤波可以有效地去除图像中的椒盐噪声，同时保持图像的边缘信息
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
median = cv2.medianBlur(pic1, 5)
image = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 双边滤波
# 双边滤波是一种非线性滤波器，它在进行滤波时会同时考虑像素的空域位置和像素值，从而在去除噪声的同时保持图像的边缘信息
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
bilateral = cv2.bilateralFilter(pic1, 15, 150, 150)
image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()


# %% [markdown]
# 边缘检测

# %%
# Roberts算子
# Roberts算子是一种简单的边缘检测算子，它通过比较图像对角方向上相邻像素的差值来检测边缘
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)
x = cv2.filter2D(image, cv2.CV_16S, kernelx)
y = cv2.filter2D(image, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# Prewitt算子
# Prewitt算子是一种常用的边缘检测算子，它通过计算图像在水平和垂直方向上的亮度变化来检测边缘
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(image, cv2.CV_16S, kernelx)
y = cv2.filter2D(image, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# Sobel 算子
# Sobel算子是一种常用的边缘检测算子，它通过计算图像在水平和垂直方向上的梯度来检测边缘
# Sobel算子相对于Prewitt算子加入了平滑处理，更加抗噪
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# Laplacian 算子
# Laplacian算子是一种二阶微分算子，它通过计算图像的二阶导数来检测边缘
# Laplacian算子检测亮度的变化速度是不是发生了变化
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
result = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
result = cv2.convertScaleAbs(result)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# Scharr 算子
# Scharr算子是Sobel算子的改进版本，它在计算梯度时使用了更高的精度
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
x = cv2.Scharr(image, cv2.CV_16S, 1, 0)
y = cv2.Scharr(image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# Canny 算子
# 一套完整的边缘检测流程
# 先降低噪声，再计算梯度，接着进行非极大值抑制，最后进行双阈值检测和边缘连接
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (3, 3), 0)
result = cv2.Canny(image, 50, 150)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# LOG算子
# 高斯拉普拉斯算子（Laplacian of Gaussian, LOG）
# 通过高斯滤波器平滑图像，然后使用拉普拉斯算子检测边缘
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (3, 3), 0)
result = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
result = cv2.convertScaleAbs(result)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()


# %% [markdown]
# 图像分割
# 把一张图像划分成若干有意义的区域

# %%
# 基于纹理的图像分割
# GrabCut 算法，提供一个大致包住目标的矩形，或者少量“确定前景 / 确定背景”的标记，算法自动估计目标的轮廓
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
mask = np.zeros(pic1.shape[:2], dtype=np.uint8)
bgdModel = np.zeros((1, 65), dtype=np.float64)
fgdModel = np.zeros((1, 65), dtype=np.float64)
# 坐标
rect = (50, 50, 1000, 700)
cv2.grabCut(pic1, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
result = pic1 * mask2[:, :, np.newaxis]
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# K-Means 聚类算法
# K-Means 聚类算法是一种经典的无监督学习算法，用来把数据自动分成 K 个类别
# 在图像处理中，根据像素的颜色或其他特征，把图像划分成若干区域。
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
rows, cols = pic1.shape[:2]
data = pic1.reshape((rows * cols, 3)).astype(np.float32)
k = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
compactness, labels, centers = cv2.kmeans(
    data,
    k,
    np.empty((data.shape[0], 1), dtype=np.int32),
    criteria,
    10,
    cv2.KMEANS_PP_CENTERS,
)
centers = centers.astype(np.uint8)
label_indices = np.asarray(labels, dtype=np.intp).ravel()
result = centers[label_indices]
result = result.reshape((rows, cols, 3))
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 均值漂移算法
# 均值漂移算法是一种无监督学习算法，它通过迭代优化来寻找数据的聚类中心
# 与 K-Means 不同，均值漂移不需要预先指定聚类的数量，而是通过数据的分布自动确定聚类的数量和位置
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
result = cv2.pyrMeanShiftFiltering(pic1, 20, 20)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 分水岭算法
# 解决多个目标互相粘连，普通二值化把它们当成一个整体的问题。
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
# 阈值化
ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 形态学操作
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# 膨胀
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# 距离变换
distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# 提取前景
ret, sure_fg = cv2.threshold(distance, 0.4 * distance.max(), 255, 0)
sure_fg = np.asarray(sure_fg, dtype=np.uint8)
# 背景
unknown = cv2.subtract(sure_bg, sure_fg)
# 标记
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
# 分水岭
markers = cv2.watershed(pic1, markers)
result = pic1.copy()
result[markers == -1] = [0, 0, 255]
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 漫水填充
# 一种从指定“种子点”开始，向周围扩展并填充相似像素的区域生长算法。
# 属于区域生长算法的一种。
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = pic1.copy()
rows, cols = image.shape[:2]
# 掩码的大小要比原图大2像素
mask = np.zeros((rows + 2, cols + 2), dtype=np.uint8)
filled_count, image, mask, rect = cv2.floodFill(
    image,
    mask,
    (30, 30),
    (0, 255, 255),
    (20, 20, 20),
    (20, 20, 20),
    cv2.FLOODFILL_FIXED_RANGE,
)
result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()


# %% [markdown]
# 频域处理

# %%
# 傅里叶变换
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft, axes=(0, 1))
result = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()

# %%
# 傅里叶逆变换
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft, axes=(0, 1))
ishift = np.fft.ifftshift(dft_shift, axes=(0, 1))
idft = cv2.idft(ishift, flags=cv2.DFT_COMPLEX_OUTPUT)
result = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()

# %%
# numpy 傅里叶变换
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
dft = np.fft.fft2(image.astype(np.float32))
dft_shift = np.fft.fftshift(dft)
result = 20 * np.log(np.abs(dft_shift))
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()
ishift = np.fft.ifftshift(dft_shift)
idft = np.fft.ifft2(ishift)
result = np.abs(idft)
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()

# %%
# 高通滤波器
# 保留图像的高频信息，去除低频信息
# 突出边缘和细节
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
dft = np.fft.fft2(np.float32(image))
dft_shift = np.fft.fftshift(dft)
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
# 创建高通滤波器
mask = np.ones((rows, cols), np.uint8)
mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0
# 应用滤波器
dft_shift = dft_shift * mask
# 傅里叶逆变换
ishift = np.fft.ifftshift(dft_shift)
idft = np.fft.ifft2(ishift)
result = np.abs(idft)
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()

# %%
# 低通滤波器
# 保留图像的低频信息，去除高频信息
# 平滑图像，去除噪声
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
dft = np.fft.fft2(np.float32(image))
dft_shift = np.fft.fftshift(dft)
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
# 创建低通滤波器
mask = np.zeros((rows, cols), np.uint8)
mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1
# 应用滤波器
dft_shift = dft_shift * mask
# 傅里叶逆变换
ishift = np.fft.ifftshift(dft_shift)
idft = np.fft.ifft2(ishift)
result = np.abs(idft)
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()


# %% [markdown]
# 形状检测

# %%
# 霍夫变换
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 50, 150)
lines = cv2.HoughLines(image, 1, np.pi / 180, 300)
if lines is None:
    raise RuntimeError("没有检测到直线")
for rho, theta in lines[:, 0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(pic1, (x1, y1), (x2, y2), (0, 255, 0), 2)
pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
plt.imshow(pic1)
plt.axis("off")
plt.show()

# %%
# 累计概率霍夫变换
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 50, 150)
lines = cv2.HoughLinesP(image, 1, np.pi / 180, 100, minLineLength=120, maxLineGap=10)
if lines is None:
    raise RuntimeError("没有检测到线段")
lines = lines.reshape(-1, 4)
for x1, y1, x2, y2 in lines:
    cv2.line(pic1, (x1, y1), (x2, y2), (0, 255, 0), 2)
result = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 霍夫圆变换
pic1 = cv2.imread("../assets/week01/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(image, 5)
circles = cv2.HoughCircles(
    image,
    cv2.HOUGH_GRADIENT,
    1.2,
    80,
    param1=120,
    param2=70,
    minRadius=20,
    maxRadius=100,
)
if circles is None:
    raise RuntimeError("没有检测到圆")
circles = np.around(circles).astype(np.uint16)
for i in circles[0, :]:
    cv2.circle(pic1, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(pic1, (i[0], i[1]), 2, (0, 0, 255), 3)
result = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()
