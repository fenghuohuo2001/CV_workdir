# -*- 练习18 -*-
"""
功能：宽度测量
作者：fenghuohuo
日期：2021年6月21日
"""
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# 用于给图片添加注释
def Img(img, text, textColor=(255, 0, 0)):
    left = 5;
    top = 5  # 设置字体在图片的位置
    textSize = 20  # 设置字体的字号
    if (isinstance(img, np.ndarray)):  # 判断是否为OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)  # 创建图片可以被注释的对象
    fontText = ImageFont.truetype("font/simhei.ttf", textSize, encoding="utf-8")  # 字体的格式
    draw.text((left, top), text, textColor, font=fontText)  # 绘制文本
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式


if __name__ == '__main__':
    # 读取原图片
    img = cv2.imread('yuantu.png')
    # 用于给图片添加中文字符
    img_text = Img(img, "yuantu.png")
    cv2.imshow("img_text", img_text)
    # 将img图像转换为灰度图
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 截取需要测量图像的区域
    recimg = GrayImage[100:200, 100:250]
    # 用于给图片添加中文字符
    recimg_text = Img(recimg, "截取测量图像")
    cv2.imshow("recimg_text ", recimg_text)
    # 对截取图像进行黑白二值化处理
    ret1, th1 = cv2.threshold(recimg, 80, 255, cv2.THRESH_BINARY)

    # 框出物体的轮廓，得到轮廓坐标和长和宽
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    # (x,y)左上角，w——矩阵宽，h——矩w阵高
    x, y, w, h = cv2.boundingRect(cnt)
    print(x, y, w, h)

    # 高斯边缘检测（白色背景黑色的物体边框）
    canny = cv2.Canny(th1, 0, 50)
    canny = 255 - cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)  # 反向
    cv2.imshow("canny", canny)

    # 将截取的图像（没有处理过的高斯边缘检测图）替换掉原图相应部分
    img[100:200, 100:250] = canny
    # 用于给图片添加中文字符
    img_1 = Img(img, "添加直线前")
    cv2.imshow("1", img_1)  # 将添加文字的e命名为“e1”并显示出来

    # 绘制直线
    cv2.line(canny, (x, y), (x, y + h), (255, 0, 255), 3)
    cv2.line(canny, (x + w, y), (x + w, y + h), (255, 0, 255), 3)

    # 将截取的图像（经过绘制直线处理过的高斯边缘检测图）替换掉原图相应部分
    img[100:200, 100:250] = canny
    # 用于给图片添加中文字符
    img_2 = Img(img, "宽度:" + str(w))
    cv2.imshow("q", img_2)  # 将添加文字的e命名为“e1”并显示出来

    # 保证程序不退出，等待键盘输入，不输入则无限等待
    cv2.waitKey(0)
    cv2.destroyAllWindows()
