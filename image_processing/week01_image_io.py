# %% [markdown]
# 第一周 Python 与图像基础


# %%
# 依赖
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# %%
# 读取和显示图像
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
cv2.imshow("test", pic1)
# 保持显示
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 用numpy查改像素
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
print(type(pic1))

# numpy读取像素
# print(pic1.item(780, 100, 0))
# print(pic1.item(780, 100, 1))
# print(pic1.item(780, 100, 2))
# 上面能用，但是不推荐了
print(pic1[780, 100])
print(int(pic1[780, 100, 0]))  # blue
print(int(pic1[780, 100, 1]))  # green
print(int(pic1[780, 100, 2]))  # red

# numpy修改像素
# pic1.itemset((780, 100, 0), 0)
# pic1.itemset((780, 100, 1), 0)
# pic1.itemset((780, 100, 2), 0)
# 以上写法已经过时
pic1[780, 100] = [0, 0, 0]
print(pic1[780, 100])

# %%
# 创建图像
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
pic2 = np.zeros(pic1.shape, np.uint8)
cv2.imshow("empty", pic2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 图像融合
pic2 = cv2.imread("../asserts/pic2.jpg")
if pic2 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
pic3 = cv2.imread("../asserts/pic3.jpg")
if pic3 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")

result = cv2.addWeighted(pic2, 1, pic3, 1, 0)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 图像属性
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
print(pic1.shape)
print(pic1.size)
print(pic1.dtype)

# %%
# 图像通道的分离
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
b, g, r = cv2.split(pic1)
cv2.imshow("blue", b)
cv2.imshow("green", g)
cv2.imshow("red", r)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# HSV颜色空间
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
hsv = cv2.cvtColor(pic1, cv2.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.axis("off")
plt.show()
h, s, v = cv2.split(hsv)
plt.subplot(1, 3, 1)
plt.imshow(h, cmap="hsv")  # 色调
plt.axis("off")
plt.title("Hue")
plt.subplot(1, 3, 2)
plt.imshow(s, cmap="gray")  # 饱和度
plt.axis("off")
plt.title("Saturation")
plt.subplot(1, 3, 3)
plt.imshow(v, cmap="gray")  # 明度
plt.axis("off")
plt.title("Value")
plt.show()

# %%
# 图像通道合并
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
b, g, r = cv2.split(pic1)
m = cv2.merge([b, g, r])
cv2.imshow("merge", m)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# 图像类型转换
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
result = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
cv2.imshow("GREY", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% [markdown]
# 图像平移
# $$
# 平移矩阵M = \begin{bmatrix}
# 1 & 0 & \Delta x \\
# 0 & 1 & \Delta y \\
# 0 & 0 & 1
# \end{bmatrix}
# $$
# 实际使用的时候只取前两行
# %%
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
m = np.array([[1, 0, 100], [0, 1, 100]], dtype=np.float32)
pic2 = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
plt.imshow(pic2)
plt.axis("off")
plt.show()

# %%
# 图像缩放
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
result = cv2.resize(pic1, (200, 100))
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()
result = cv2.resize(pic1, None, fx=0.5, fy=0.5)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 图像旋转
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
rows, cols = image.shape[:2]
m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 60, 1)
pic2 = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
plt.imshow(pic2)
plt.axis("off")
plt.show()

# %%
# 图像翻转
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
pic2 = cv2.flip(image, 1)  # 1表示水平翻转，0表示垂直翻转，-1表示水平和垂直同时翻转
plt.imshow(pic2)
plt.axis("off")
plt.show()

# %%
# 图像仿射
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
pts1 = np.array([[50, 50], [200, 50], [50, 200]], dtype=np.float32)
pts2 = np.array([[10, 100], [200, 50], [100, 250]], dtype=np.float32)
m = cv2.getAffineTransform(pts1, pts2)
pic2 = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
plt.imshow(pic2)
plt.axis("off")
plt.show()

# %%
# 图像透视
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
pts1 = np.array([[50, 50], [200, 50], [50, 200], [200, 200]], dtype=np.float32)
pts2 = np.array([[10, 100], [200, 50], [100, 250], [250, 250]], dtype=np.float32)
m = cv2.getPerspectiveTransform(pts1, pts2)
pic2 = cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]))
plt.imshow(pic2)
plt.axis("off")
plt.show()

# %%
# 图像量化
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
# 将图像转换为8位无符号整数
image = np.uint8(image)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 向上取样
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
result = cv2.pyrUp(image)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 向下取样
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
result = cv2.pyrDown(image)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 图像灰度化
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 二进制阈值化
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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

# %%
# 图像腐蚀
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
# 原图-开区间
# 提取小尺寸的亮点区域
pic1 = cv2.imread("../asserts/pic1.jpg")
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
# 原图-闭区间
# 提取小尺寸的暗点区域
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
result = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 灰度直方图
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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


pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
result = auto_color_balance(pic1, radius=15, alpha=2.5, clip_percent=1.0)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 均值滤波
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
blur = cv2.blur(pic1, (5, 5))
image = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 方框滤波
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
box = cv2.boxFilter(pic1, -1, (3, 3), normalize=False)
image = cv2.cvtColor(box, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 高斯滤波
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
gaussian = cv2.GaussianBlur(pic1, (5, 5), 0)
image = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 中值滤波
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
median = cv2.medianBlur(pic1, 5)
image = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# 双边滤波
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
bilateral = cv2.bilateralFilter(pic1, 15, 150, 150)
image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# Roberts算子
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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

# %%
# 基于纹理的图像分割
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
rows, cols = pic1.shape[:2]
data = pic1.reshape((rows * cols, 3)).astype(np.float32)
k = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
compactness, labels, centers = cv2.kmeans(
    data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
)
centers = np.uint8(centers)
result = centers[labels.flatten()]
result = result.reshape((rows, cols, 3))
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 均值漂移算法
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
result = cv2.pyrMeanShiftFiltering(pic1, 20, 20)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

# %%
# 分水岭算法
pic1 = cv2.imread("../asserts/pic1.jpg")
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
sure_fg = np.uint8(sure_fg)
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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

# %%
# 傅里叶变换
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft, axes=(0, 1))
result = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()

# %%
# 傅里叶逆变换
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft, axes=(0, 1))
ishift = np.fft.ifftshift(dft_shift, axes=(0, 1))
idft = cv2.idft(ishift, flags=cv2.DFT_COMPLEX_OUTPUT)
result = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()

# %%
# numpy 傅里叶变换
pic1 = cv2.imread("../asserts/pic1.jpg")
if pic1 is None:
    raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
dft = np.fft.fft2(np.float32(image))
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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
pic1 = cv2.imread("../asserts/pic1.jpg")
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

# %%
