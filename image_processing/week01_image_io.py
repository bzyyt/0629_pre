# %% [markdown]
# 第一周 Python 与图像基础


# %%
# 依赖
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
# %%
