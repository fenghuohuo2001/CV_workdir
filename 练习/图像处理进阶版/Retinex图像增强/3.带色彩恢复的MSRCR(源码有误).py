# -*- coding: UTF-8 -*-
"""
@Project : pythonProject
@File    : 3.带色彩恢复的MSR.py
@IDE     : PyCharm
@Author  : FengHuohuo
@Date    : 2022/7/2 16:43
@Function：
"""
import cv2
import numpy as np

# MSRCR
def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


# simple color balance
def simple_color_balance(input_img, s1, s2):
    h, w = input_img.shape[:2]
    out_img = input_img.copy()
    sort_img = input_img.copy()
    one_dim_array = sort_img.flatten()  # 转化为一维数组
    sort_array = sorted(one_dim_array)  # 对一维数组按升序排序

    num = h * w
    print(num)
    # 取最小值部分像素值的索引
    per1 = int((h * w) * s1/100)
    print(per1)
    minvalue = sort_array[per1]

    # 取最大值部分像素值的索引
    per2 = int((h * w) * s2/100)
    print(per2)
    maxvalue = sort_array[(h * w) - 1 - per2]

    # 实施简单白平衡算法
    if (maxvalue <= minvalue):  # 若取到的最大值索引<最小值索引
        for i in range(h):
            for j in range(w):
                out_img[i, j] = maxvalue    # 将图像所有像素均设为像素最大值
    else:                       # 若是正常图像取值
        scale = 255.0 / (maxvalue - minvalue)   # 线性拉伸参数
        for m in range(h):
            for n in range(w):
                if (input_img[m, n] < minvalue):    # 将小于最小像素设为0
                    out_img[m, n] = 0
                elif (input_img[m, n] > maxvalue):  # 将大于最大像素设为255
                    out_img[m, n] = 255
                else:                               # 中间段像素进行拉伸
                    out_img[m, n] = scale * (input_img[m, n] - minvalue)  # 映射中间段的图像像素

    out_img = cv2.convertScaleAbs(out_img)      # 将得到的有些负值取绝对值得到正数，并将数据转化到0-255之间
    return out_img

# s1,s2 是去掉最小值和最大值的(量程100)，即去s1~s2中间部分
def MSRCR(img, scales, s1, s2):
    h, w = img.shape[:2]
    MSRCR_Out = img.copy()
    scales_size = 3

    B, G, R = cv2.split(img)
    log_R = np.zeros((h, w), dtype=np.float32)
    I = np.zeros((h, w), dtype=np.float32)

    I = np.add(B, G, R)
    I = replaceZeroes(I)

    for j in range(3):
        img[:, :, j] = replaceZeroes(img[:, :, j])
        for i in range(0, scales_size):
            L_blur = cv2.GaussianBlur(img[:, :, j], (scales[i], scales[i]), 0)
            L_blur = replaceZeroes(L_blur)
            dst_img = cv2.log(img[:, :, j]/255.0)   # 注意一下 最好用x = cv2.log（y）
            dst_Lblur = cv2.log(L_blur/255.0)
            dst_Ixl = cv2.multiply(dst_img, dst_Lblur)
            log_R += cv2.subtract(dst_img, dst_Ixl)
        MSR = log_R / 3.0
        MSRCR = 46 * (cv2.log(125.0 * img[:, :, j]/255.0) - cv2.log(I/255.0))
        MSRCR = cv2.normalize(MSRCR, 0, 255, cv2.NORM_MINMAX)
        MSRCR_Out[:, :, j] = simple_color_balance(MSRCR, s1, s2)
    return MSRCR_Out

if __name__ == "__main__":
    # image = cv2.imread("src.jpg")
    image = cv2.imread("rabbit.png")
    # image = 255 - image
    cv2.imshow("MRC", image)

    h, w = image.shape[:2]
    scales = [3, 5, 9]
    low = 10
    high = 110
    MSRCR_Out = MSRCR(image, scales, low, high)
    cv2.imshow("MSRCR", MSRCR_Out)
    cv2.waitKey(0)
    cv2.destroyWindow()