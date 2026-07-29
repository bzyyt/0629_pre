import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

for i in range(4):
    # 读取图片
    image = cv2.imread("asserts/week2_work/pic" + str(i + 1) + ".jpg")
    if image is None:
        raise FileNotFoundError("图片读取失败，请检查当前工作目录和图片路径")
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 灰度化
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # canny边缘检测
    edges = cv2.Canny(image, 80, 160)

    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel)

    # 外轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取大轮廓
    package_contour = 0
    for contour in contours:
        # area = cv2.contourArea(contour)
        # if area > 100:
        #     package_contours.append(contour)
        # x, y, w, h = cv2.boundingRect(contour)
        # if w > 250 and h > 150:
        #     package_contours.append(contour)
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        short = min(rect_height, rect_width)
        long = max(rect_width, rect_height)
        if 140 < short < 400 and 240 < long < 500:
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(result, [box], 0, (0, 0, 255), 3)
            package_contour += 1
    print("轮廓数量：", package_contour)

    # 绘制轮廓
    plt.imshow(result)
    plt.axis("off")
    plt.savefig("results/week2_cv/pic" + str(i + 1) + "_result.jpg")
    # plt.show()
