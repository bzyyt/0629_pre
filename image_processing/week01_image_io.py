import cv2
import matplotlib.pyplot as plt
import numpy as np

# matplotlib 中文字体问题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

for i in range(3):
    # 读取图像
    image = cv2.imread("asserts/pic" + str(i + 1) + ".jpg")
    if image is None:
        raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
    plt.figure(figsize=(10, 10))
    # 显示原图
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("原图")
    # 灰度图
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("results/week1_image_io/pic" + str(i + 1) + "_grey.jpg", grey)
    plt.subplot(2, 2, 2)
    plt.imshow(grey, cmap="gray")
    plt.title("灰度图")
    # HSV图
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("results/week1_image_io/pic" + str(i + 1) + "_hsv.jpg", hsv)
    plt.subplot(2, 2, 3)
    plt.imshow(hsv)
    plt.title("HSV图")
    # 缩小图
    small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite("results/week1_image_io/pic" + str(i + 1) + "_small.jpg", small)
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    plt.title("缩小图")
    plt.tight_layout()
    plt.savefig("results/week1_image_io/pic" + str(i + 1) + "_result.jpg")
    plt.show()
    # 输出数据
    print("图片" + str(i + 1) + "的尺寸为：" + str(image.shape[:2]))
    print("图片" + str(i + 1) + "的通道数为：" + str(image.shape[2]))
    print(
        "图片"
        + str(i + 1)
        + "的像素值范围为："
        + str(np.min(image))
        + " ~ "
        + str(np.max(image))
    )
