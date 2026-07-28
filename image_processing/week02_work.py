import cv2
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

for i in range(1):
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

    # 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    edges = cv2.dilate(edges, kernel)

    # 外轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取大轮廓
    package_contours = []
    for contour in contours:
        # area = cv2.contourArea(contour)
        # if area > 100:
        #     package_contours.append(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if w > 250 and h > 150:
            package_contours.append(contour)
    print("轮廓数量：", len(package_contours))

    # 绘制轮廓
    cv2.drawContours(result, package_contours, -1, (0, 0, 255), 3)
    plt.imshow(result)
    plt.axis("off")
    plt.show()
